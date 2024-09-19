
from transformers import Trainer

from transformers.training_args import OptimizerNames
from typing import Dict, Union, Any
import torch
import torch.nn as nn

from transformers.utils import is_sagemaker_mp_enabled
if is_sagemaker_mp_enabled():
    from transformers.trainer_pt_utils import smp_forward_backward

from transformers import is_apex_available
if is_apex_available():
    from apex import amp


class CustomTrainer(Trainer):
    
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:

        """
        Copied from default Trainer (transformers 4.42.3) and add additional logging
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True) # <- modified to return outputs

        if 'loss_terms' in outputs:
            for k in outputs['loss_terms']:
                outputs['loss_terms'][k] = outputs['loss_terms'][k].item() 
            self.log(outputs['loss_terms'])

        del inputs

        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learnign rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss, **kwargs)

        return loss.detach() / self.args.gradient_accumulation_steps

@torch.no_grad()
def compute_metrics_for_loss_term_logging(predictions):
    return {k: v.mean() for k, v in predictions.predictions.items()}