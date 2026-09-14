from typing import Optional, Union, Tuple, Dict, List, Any
from functools import reduce
import torch
from ..._typedefs import *
from ..modules.mlp_block import MLPBlock
from ..modules.embeddings import FourierEmbeddings
from ..utils import utils
from ..wrapper.x_dependent import XDependentSequential
from ..._utils import apply_recursively

class TimeMLP(torch.nn.Module):

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        hidden_dims: IntParams,
        *,
        activation_function: torch.nn.Module = torch.nn.SiLU(),
        time_embedding_size: Optional[int] = None,
        zero_out: bool = False,
        simplified: bool = True,
        project_input: bool = False,
        condition_input_dims: Optional[Union[int, Dict[str, Any], List[int]]] = None,
        condition_hidden_dims: Optional[int] = None,
        condition_concat: bool = False,
        condition_projection: bool = False,
        condition_projection_module: Optional[torch.nn.Module] = None,
        pretrained_condition_module: bool = False,
        n_classes: Optional[int] = None,
        class_drop_prob: float = 0.,
    ):
        super().__init__()
        self._input_dims = input_dims
        self._output_dims = output_dims
        self._hidden_dims = hidden_dims
        self._simplified = simplified
        self._project_input = project_input

        # Condition Config
        self._condition_input_dims = condition_input_dims
        self._condition_hidden_dims = hidden_dims if condition_hidden_dims is None else condition_hidden_dims
        self._condition_concat = condition_concat
        self._condition_projection = condition_projection
        self._pretrained_condition_module = pretrained_condition_module

        # Global Config
        self._time_dependent = time_embedding_size is not None
        self._time_embedding_size = time_embedding_size
        self._n_classes = n_classes
        self._class_drop_prob = class_drop_prob
        self._has_condition = (n_classes is not None or condition_input_dims is not None) and (condition_projection or condition_concat or self._simplified)

        self._internal_dim = None
        if not self._simplified:
            if self._time_dependent:
                self._internal_dim = self._time_embedding_size * 4
            else:
                self._internal_dim = hidden_dims[0] if isinstance(hidden_dims, (list, tuple)) else hidden_dims

            if self._project_input:
                self.input_projector = torch.nn.Linear(self._input_dims, self._internal_dim)

        self._time_emb_dim = 0
        if self._time_dependent:
            if self._simplified:
                self._time_emb_dim = 1 # Scalar concatenation
            else:
                self.time_embedder = FourierEmbeddings(time_embedding_size)
                self.embedding_projection = MLPBlock(
                    input_dims=time_embedding_size,
                    output_dims=self._internal_dim, # Match D for addition
                    hidden_dims=[self._internal_dim],
                    activation_function=[torch.nn.SiLU()],
                    output_activation=False,
                )
                self._time_emb_dim = self._internal_dim

        self._class_emb_dim = 0
        if self._n_classes is not None:
            self._null_class_idx = self._n_classes
            if self._simplified:
                self._class_emb_dim = self._n_classes + 1 if self._class_drop_prob > 0 else self._n_classes
            else:
                self.class_embedder = torch.nn.Embedding(
                    self._n_classes + 1 if self._class_drop_prob > 0 else self._n_classes,
                    self._internal_dim,
                )
                self._class_emb_dim = self._internal_dim

        self.condition_projection_block = None
        self._condition_output_dim = 0
        self._bypass_capable = False

        raw_cond_size = 0
        if self._has_condition and self._condition_input_dims is not None:
            if isinstance(self._condition_input_dims, int):
                raw_cond_size = self._condition_input_dims
            elif isinstance(self._condition_input_dims, (dict, list, tuple)):
                def get_size(obj):
                    if isinstance(obj, int): return obj
                    if isinstance(obj, (list, tuple)): return reduce(lambda a, b: a * b, obj)
                    if isinstance(obj, torch.Tensor): return obj.prod().int().item()
                    return 0

                if isinstance(self._condition_input_dims, dict):
                    raw_cond_size = sum(get_size(v) for v in self._condition_input_dims.values())
                else:
                    raw_cond_size = sum(get_size(v) for v in self._condition_input_dims)

        if self._simplified:
            self._condition_output_dim = raw_cond_size
            self._bypass_capable = True
        else:
            self._custom_condition_module = False
            if self._has_condition and self._condition_projection:
                if condition_projection_module is not None:
                    self.condition_projection_block = XDependentSequential(condition_projection_module)
                    # Calculate output size
                    out_shape = utils.calculate_output_size(self.condition_projection_block, input_dims=self._condition_input_dims)
                    self._condition_output_dim = torch.tensor(out_shape).prod().item()
                    self._custom_condition_module = True

                    if raw_cond_size == self._condition_output_dim:
                        self._bypass_capable = True

                elif self._has_condition:
                    self.condition_projection_block = MLPBlock(
                        input_dims=raw_cond_size,
                        output_dims=self._internal_dim,
                        hidden_dims=self._condition_hidden_dims,
                        activation_function=[activation_function],
                        output_activation=False,
                        zero_out=zero_out,
                    )
                    self._condition_output_dim = self._internal_dim
                    self._bypass_capable = (self._condition_input_dims == self._internal_dim)
                else:
                    raise ValueError("Advanced mode requires 'condition_projection_module' for Dict/List inputs.")

            elif self._condition_concat:
                self._condition_output_dim = raw_cond_size

        if self._simplified:
            total_input_dim = self._input_dims + self._time_emb_dim + self._class_emb_dim + self._condition_output_dim
        else:
            if self._project_input:
                total_input_dim = self._internal_dim
            else:
                total_input_dim = self._input_dims + self._time_emb_dim + self._class_emb_dim

            if self._condition_concat or self._condition_projection:
                total_input_dim += self._condition_output_dim

        self.fc = MLPBlock(
            input_dims=total_input_dim,
            output_dims=self._output_dims,
            hidden_dims=hidden_dims,
            activation_function=[activation_function],
            output_activation=False,
            zero_out=zero_out,
        )

    def _prepare_time(self, t: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        if t is None: return None

        if self._simplified:
            if t.ndim == 1: t = t.unsqueeze(-1)
            return t
        else:
            assert self._time_dependent, "Model is not time-dependent, but time has provided as input"
            emb = self.time_embedder(t)
            emb = self.embedding_projection(emb)
            return emb

    def _prepare_labels(
        self, y: Optional[torch.Tensor], batch_size: int, device: torch.device
    ) -> Optional[torch.Tensor]:
        if self._n_classes is None: return None

        if y is None:
            if self._class_drop_prob > 0:
                y = torch.full((batch_size,), self._null_class_idx, device=device, dtype=torch.long)
            else:
                raise ValueError(f"Model requires labels.")
        else:
            if self.training and self._class_drop_prob > 0:
                drop_mask = torch.bernoulli(torch.full(y.shape, self._class_drop_prob, device=y.device)).bool()
                y = torch.where(drop_mask, self._null_class_idx, y)

        if self._simplified:
            vocab = self._n_classes + 1 if self._class_drop_prob > 0 else self._n_classes
            return torch.nn.functional.one_hot(y, num_classes=vocab).float()
        else:
            return self.class_embedder(y)

    def prepare_conditioning(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        c: Optional[Union[torch.Tensor, Tuple, Dict]] = None,
        y: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:

        has_condition_inputs = (c is not None) or (y is not None) or (z is not None)
        is_unconditional_valid = (self._n_classes is not None and self._class_drop_prob > 0)

        if self._has_condition and not has_condition_inputs and not is_unconditional_valid:
             raise ValueError("Conditioning enabled but no inputs provided.")

        t_emb = self._prepare_time(t)
        y_emb = self._prepare_labels(y, batch_size=x.shape[0], device=x.device)
        c_emb = None

        if z is not None:
            c_emb = z
        elif c is not None:
            if self._simplified:
                c_components = apply_recursively(c, lambda x: x.flatten(1))
                if c_components is not None:
                    c_emb = torch.cat(
                        tuple(c_components.values())
                        if isinstance(c_components, dict)
                        else tuple(c_components),
                        dim=-1
                    )if not isinstance(c_components, torch.Tensor) else c_components
            else:
                if self._condition_projection and self._custom_condition_module:
                    if isinstance(c, dict): c_emb = self.condition_projection_block(c)
                    elif isinstance(c, (tuple, list)): c_emb = self.condition_projection_block(*c)
                    else: c_emb = self.condition_projection_block(c)
                else:
                    c_components = apply_recursively(c, lambda x: x.flatten(1))
                    c_emb = torch.cat(
                        tuple(c_components.values())
                        if isinstance(c_components, dict)
                        else tuple(c_components),
                        dim=-1
                    ) if not isinstance(c_components, torch.Tensor) else c_components
                    if self._condition_projection:
                        c_emb = self.condition_projection_block(c_emb)

            if c_emb is not None and c_emb.ndim > 2:
                c_emb = c_emb.flatten(1)

        return t_emb, y_emb, c_emb

    def forward(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        c: Optional[Union[Dict[str, torch.Tensor], torch.Tensor, Tuple]] = None,
    ) -> torch.Tensor:

        condition, labels, projection = None, None, None

        if isinstance(c, dict):
            condition = c.get("condition")
            labels = c.get("labels")
            projection = c.get("projection")
            if condition is None and labels is None and projection is None:
                condition = c
        elif isinstance(c, tuple):
            condition = c
        elif isinstance(c, torch.Tensor):
            # Safe bypass logic
            flat_dim = c.shape[-1]
            target_dim = self._condition_output_dim
            if self._bypass_capable and flat_dim == target_dim:
                projection = c
            else:
                condition = c

        t_emb, y_emb, c_emb = self.prepare_conditioning(x, t, c=condition, y=labels, z=projection)

        if self._simplified:
            x_in = [x]
            if c_emb is not None: x_in.append(c_emb)
            if t_emb is not None: x_in.append(t_emb)
            if y_emb is not None: x_in.append(y_emb)
            x_input = torch.cat(x_in, dim=-1)
        else:
            h = x
            if self._project_input:
                h = self.input_projector(h)
                if t_emb is not None: h = h + t_emb
                if y_emb is not None: h = h + y_emb
            else:
                raw_in = [h]
                if t_emb is not None: raw_in.append(t_emb)
                if y_emb is not None: raw_in.append(y_emb)
                h = torch.cat(raw_in, dim=-1)

            if c_emb is not None:
                h = torch.cat((h, c_emb), dim=-1)
            x_input = h

        return self.fc(x_input)

    def train(self, mode=True):
        super().train(mode)
        if mode and self._pretrained_condition_module and not self._simplified:
            if isinstance(self.condition_projection_block, torch.nn.Module):
                self.condition_projection_block.eval()
                for param in self.condition_projection_block.parameters():
                    param.requires_grad = False
        return self
