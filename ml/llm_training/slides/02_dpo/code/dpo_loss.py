import torch.nn.functional as F

def dpo_loss(pi_logps, ref_logps, yw, yl, beta):
    # log-probs of the whole chosen / rejected sequences
    pi_w,  pi_l  = pi_logps[yw],  pi_logps[yl]
    ref_w, ref_l = ref_logps[yw], ref_logps[yl]

    pi_logratio  = pi_w  - pi_l
    ref_logratio = ref_w - ref_l

    # binary cross-entropy on the implicit-reward margin
    loss = -F.logsigmoid(beta * (pi_logratio - ref_logratio))

    # implicit rewards (for logging only)
    rewards = beta * (pi_logps - ref_logps).detach()
    return loss, rewards
