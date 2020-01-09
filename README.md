# ParsNet
Weakly Supervised Deep Learning Approach in Streaming Environments. Proceedings of The 2019 IEEE International Conference on Big Data (IEEE Big Data 2019).

# Abstract
The feasibility of existing data stream algorithms is often hindered by the weakly supervised condition of data streams. A self-evolving deep neural network, namely Parsimonious Network (ParsNet), is proposed as a solution to various weakly-supervised data stream problems. A self-labelling strategy with hedge (SLASH) is proposed in which its auto-correction mechanism copes with \textit{the accumulation of mistakes} significantly affecting the model's generalization. ParsNet is developed from a closed-loop configuration of the self-evolving generative and discriminative training processes exploiting shared parameters in which its structure flexibly grows and shrinks to overcome the issue of concept drift with/without labels. The numerical evaluation has been performed under two challenging problems, namely sporadic access to ground truth and infinitely delayed access to the ground truth. Our numerical study shows the advantage of ParsNet with a substantial margin from its counterparts in the high-dimensional data streams and infinite delay simulation protocol. To support the reproducible research initiative, the source code of ParsNet along with supplementary materials are made available at https://bit.ly/2qNW7p4.

# Citation
If you use this code, please cite:\\
@article{pratama2019weakly,
  title={Weakly Supervised Deep Learning Approach in Streaming Environments},
  author={Pratama, Mahardhika and Ashfahani, Andri and Hady, Mohamad Abdul},
  journal={arXiv preprint arXiv:1911.00847},
  year={2019}
}
