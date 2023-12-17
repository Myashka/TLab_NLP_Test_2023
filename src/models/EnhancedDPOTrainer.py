from typing import Literal, Tuple, Dict, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from trl import DPOTrainer


class EnhancedDPOTrainer(DPOTrainer):
    def __init__(
        self,
        alpha: float = None,
        label_smoothing: float = 0,
        annealing: bool = False,
        loss_type: Literal[
            "sigmoid", "hinge", "fkl", "jsd", "alpha_divergence", "ipo"
        ] = "sigmoid",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.loss_type = loss_type
        self.annealing = annealing
        if self.annealing:
            if self.loss_type == "ipo":
                self.beta = 1.0
                self.beta_increment = (
                    0.01 - 1.0
                ) / self.args.max_steps  # Декремент для IPO
            if self.loss_type == "sigmoid":
                self.beta = 0.
                self.beta_increment = (
                    1.0
                ) / self.args.max_steps  # Инкремент для sigmoid

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        loss = super().training_step(model, inputs)
        if self.annealing:
            self.beta += self.beta_increment
        return loss

    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_free: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
            beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
            reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """

        if self.loss_type in ["sigmoid", "hinge", "ipo"]:
            pi_logratios = policy_chosen_logps - policy_rejected_logps
            if reference_free:
                ref_logratios = 0
            else:
                ref_logratios = reference_chosen_logps - reference_rejected_logps
            logits = pi_logratios - ref_logratios
            if self.loss_type == "sigmoid":
                losses = (
                    -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                    - F.logsigmoid(-self.beta * logits) * self.label_smoothing
                )
            elif self.loss_type == "hinge":
                losses = torch.relu(1 - self.beta * logits)
            elif self.loss_type == "ipo":
                # eqn (17) of the paper where beta is the regularization parameter for the IPO loss, denoted by tau in the paper.
                losses = (logits - 1 / (2 * self.beta)) ** 2

        elif self.loss_type == "jsd":
            pi_logratios = policy_chosen_logps - policy_rejected_logps
            chosen_denominator = torch.logaddexp(
                policy_chosen_logps, reference_chosen_logps
            )
            rejected_denominator = torch.logaddexp(
                policy_rejected_logps, reference_rejected_logps
            )
            logits = pi_logratios - chosen_denominator + rejected_denominator
            losses = -F.logsigmoid(self.beta * logits)

        elif self.loss_type == "fkl":
            rejected_diff = reference_rejected_logps - policy_rejected_logps
            chosen_diff = reference_chosen_logps - policy_chosen_logps
            logits = torch.exp(rejected_diff) - torch.exp(chosen_diff)
            losses = -F.logsigmoid(self.beta * logits)

        elif self.loss_type == "alpha_divergence":
            if not hasattr(self, "alpha"):
                raise ValueError("Alpha parameter for alpha-divergence is not defined.")
            ratio_rejected = reference_rejected_logps - policy_rejected_logps
            ratio_chosen = reference_chosen_logps - policy_chosen_logps

            ratio_rejected = torch.exp(ratio_rejected * self.alpha)
            ratio_chosen = torch.exp(ratio_chosen * self.alpha)

            logits = ratio_rejected - ratio_chosen
            losses = -F.logsigmoid((self.beta / self.alpha) * logits)
        elif self.loss_type == "kto":
            # eqn (7) of the HALOs paper
            chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
            rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)

            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
            # As described in the KTO report, the KL term for chosen (rejected) is estimated using the rejected (chosen) half.
            losses = torch.cat(
                (
                    1 - F.sigmoid(self.beta * (chosen_logratios - rejected_KL)),
                    1 - F.sigmoid(self.beta * (chosen_KL - rejected_logratios)),
                ),
                0,
            )
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'jsd', 'fkl', 'alpha_divergence', 'ipo']"
            )

        chosen_rewards = (
            self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        )
        rejected_rewards = (
            self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
        )

        return losses, chosen_rewards, rejected_rewards
