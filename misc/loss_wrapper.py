import torch
import misc.utils as utils
from misc.rewards import init_scorer, get_self_critical_reward

class LossWrapper(torch.nn.Module):
    def __init__(self, model, opt):
        super(LossWrapper, self).__init__()
        self.opt = opt
        self.model = model
        if opt.label_smoothing > 0:
            self.crit = utils.LabelSmoothing(smoothing=opt.label_smoothing)
        else:
            self.crit = utils.LanguageModelCriterion()
        self.cp_crit = torch.nn.BCELoss()
        self.rl_crit = utils.RewardCriterion()

    def forward(self, fc_feats, att_feats, labels, masks, att_masks, gts, gt_indices,
                sc_flag, visual_concepts = None):
        out = {}
        if not sc_flag:
            if visual_concepts is None:
                outputs = self.model(fc_feats, att_feats, labels, att_masks)
                loss = self.crit(outputs, labels[:, 1:], masks[:, 1:])
            else:
                outputs, concept_output = self.model(fc_feats, att_feats, labels, att_masks)
                loss1 = self.crit(outputs, labels[:, 1:], masks[:, 1:])
                loss2 = self.cp_crit(concept_output,visual_concepts.float())
                loss = loss1 + self.opt.concept_lambda * loss2
        else:
            gen_result, sample_logprobs = self.model(fc_feats, att_feats, att_masks, opt={'sample_max':0}, mode='sample')
            gts = [gts[_] for _ in gt_indices.tolist()]
            reward = get_self_critical_reward(self.model, fc_feats, att_feats, att_masks, gts, gen_result, self.opt)
            reward = torch.from_numpy(reward).float().to(gen_result.device)
            loss = self.rl_crit(sample_logprobs, gen_result.data, reward)
            out['reward'] = reward[:,0].mean()
        out['loss'] = loss
        return out
