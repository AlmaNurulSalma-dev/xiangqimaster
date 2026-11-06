# Benchmarking Minimax, Imitation, Self-Play Reinforcement Learning, and Monte Carlo Tree Search in a Multi-Agent Xiangqi (象棋) League

@META Alma Nurul Salma (诺艾玛)
@META Nanjing Xiaozhuang University (南京晓庄学院) · Student ID: L25020007
@META almanurulsalma@gmail.com · Research proposal · 6 November 2026

@ABSTRACT
Four families of artificial intelligence have each produced a landmark result in board games: adversarial tree search, supervised imitation of human experts, self-play reinforcement learning, and Monte Carlo tree search over a learned network. They have never been compared with one another. Each published demonstration uses a different game, architecture and amount of computation, so a practitioner cannot tell which approach actually plays best, nor what limits the others. This proposal sets out a controlled study that removes those confounds. All four paradigms are implemented on one domain, Xiangqi (Chinese chess), over a single shared rule engine, a 2,086-move action space, one policy-value network and one Elo league, so the only variable is the learning algorithm. The study ranks the paradigms with confidence intervals, measures whether imitation pretraining helps or hinders the self-play that follows, and ablates four levers to identify what bounds strength. A complete, tested implementation with preliminary measurements already exists, making this a rigorous formalisation rather than a speculative build. The contributions are a reusable evaluation league, the first like-for-like paradigm comparison for Xiangqi, and reproducibility guidance for self-play reinforcement learning.

@KEYWORDS Xiangqi, Chinese chess, game artificial intelligence, Monte Carlo tree search, imitation learning, Proximal Policy Optimization, Elo rating, reproducibility

---

## I. INTRODUCTION

### A. Background

The dominant recipe for superhuman board-game play was established on Go by AlphaGo, which combined supervised imitation of expert moves, self-play reinforcement learning, and Monte Carlo tree search [1]. AlphaGo Zero removed the human-data stage [2], and AlphaZero generalised the algorithm to chess, shogi and Go [3]. Each component has its own lineage: adversarial search was formalised by Shannon [4], made practical by the alpha-beta analysis of Knuth and Moore [5], and reached the world champion in Deep Blue [6]; behaviour cloning began with imitating a human driver [10]; and the tree search was introduced by Coulom [7] and given its selection rule by Kocsis and Szepesvári [8].

### B. The problem, and where it comes from

**Problem.** The four paradigms are never compared against one another. They are compared against humans, or against the previous state of the art in whichever domain the authors chose. Nobody can therefore say which plays a given game best, or what limits each, because no two published results share a domain, an architecture and a training budget.

This is not a suspicion of the author's but a documented methodological failure in the field. Henderson *et al.* showed that reported differences between deep reinforcement-learning algorithms are often attributable to implementation details, seeds and evaluation choices rather than to the algorithms compared, and argued for controlled comparisons in a shared codebase with reported variance [15]. The problem therefore originates in the published literature itself, and this project applies that recommendation where it has not been tried.

The domain is Xiangqi (象棋, Chinese chess): a culturally major but academically under-served game played on a 9 x 10 board, with a cannon that captures by jumping a screen, a horse that can be blocked, and the unusual rule that stalemate is a loss. Reported figures place its branching factor near thirty-eight against roughly thirty-five for chess, and its game-tree complexity near 10^150 against roughly 10^123 [18].

**Solution.** Hold the environment, representation, action space, network and evaluation protocol fixed, and swap only the learning paradigm. A like-for-like ranking then isolates the effect of the paradigm, and targeted ablations attribute the remainder to search, capacity, data or algorithm.

### C. Research questions and objectives

**RQ1 (ranking).** With domain, representation, architecture and protocol held constant, how do the four paradigms rank on one Elo scale?

**RQ2 (imitation effect).** Does initialising self-play from an imitation-learned policy improve the final agent or constrain it? The literature motivates both: imitation gives a strong starting policy [12], but imposes the demonstrator's state distribution, a liability once the policy departs from it [11].

**RQ3 (binding constraint).** When an agent stops improving, is the cause play-time search, model capacity, the data or teacher, or the learning algorithm?

The objectives are **O1** a rules-correct environment with a fixed action space and shared encoding; **O2** the four paradigms over it, plus a random floor and an engine reference; **O3** an Elo league with 95% confidence intervals and a held-out move-prediction metric; **O4** ablations isolating search, capacity, teacher and training length; and **O5** release of the environment, datasets, checkpoints and demonstrator for reuse.

---

## II. RELATED WORK AND THE RESEARCH GAP

Each source is identified with the specific point it contributes.

**Adversarial search.** Shannon formalised minimax [4], Knuth and Moore quantified alpha-beta pruning [5], and Campbell *et al.* documented Deep Blue [6]. *Useful point:* it needs no data and no training, giving a learning-free control condition for every learned agent.

**Imitation learning.** Pomerleau demonstrated behaviour cloning [10], Hussein *et al.* survey its failure modes [12], and Ross *et al.* formalised the compounding error from distribution shift [11]. *Useful point:* theory predicts an imitator is strong where demonstrations are dense and degrades where they thin, giving RQ2 a falsifiable prediction rather than a guess.

**Self-play reinforcement learning.** Schulman *et al.* introduced Proximal Policy Optimization, chosen here for stability [13], and Vinyals *et al.* reached grandmaster level by self-play in a far less structured domain [14]. *Useful point:* it needs no human data but is sample-hungry, making it the paradigm most sensitive to a limited training budget.

**Tree search with a learned network.** Coulom introduced the search [7], Kocsis and Szepesvári supplied the selection rule [8], and Browne *et al.* survey the design space [9]. *Useful point:* this is the strongest known recipe, and supplies the rule the search agent here implements.

**Reproducibility.** Henderson *et al.* showed policy-gradient results vary sharply with seeds and implementation [15]; Tian *et al.* reimplemented AlphaZero openly and reported the resources it required [16]. *Useful point:* these justify interval-based evaluation, a shared codebase, and reporting the budget beside every result.

**Xiangqi-specific work.** Li *et al.* introduced JiangJun, combining tree search with population methods over more than ten thousand human games to address non-transitivity [17]; Yilmaz *et al.* describe an AlphaZero-style player and report the complexity figures above [18]; Chen *et al.* use Xiangqi to test spatial reasoning in language models [19]. *Useful point:* these establish Xiangqi as a legitimate research venue, but each optimises a single pipeline and **none compares paradigms**.

**Research gap.** {{t:gap}} makes the position explicit. No controlled, Elo-anchored comparison of classical search, self-play reinforcement learning, imitation-initialised reinforcement learning and AlphaZero-style tree search exists for Xiangqi. Both the direction of the imitation effect and the binding constraint on strength are therefore unquantified for this game. This is primarily an evaluation and methodology gap, with secondary performance and application dimensions.

@TABLE gap | Where prior work leaves the gap this study fills
| Work | Xiangqi | Cross-paradigm comparison | Analyses what limits strength |
|---|---|---|---|
| AlphaGo family [1]–[3] | no | no | no |
| MCTS survey [9] | n/a | n/a | n/a |
| Proximal Policy Optimization [13] | no | partial | no |
| Reproducibility [15] | no | no | partial |
| AlphaStar [14] | no | no | no |
| JiangJun [17] | yes | no | no |
| Yilmaz et al. [18] | yes | no | no |
| Xiangqi-R1 [19] | yes | no | no |
| **This study** | **yes** | **yes** | **yes** |

---

## III. DESIGN DECISIONS AND THEIR RATIONALE

Five decisions determine the shape of the study. Each is recorded with its reason and the cost it carries, so the design can be challenged on its merits ({{t:decisions}}).

**D1, the domain.** Xiangqi is harder to search than chess [18] yet remains perfect-information and zero-sum, so classical theory applies unmodified; it is under-studied; and open data and an open engine make rigorous evaluation possible without proprietary resources.

**D2, the four paradigms.** They are the four historically distinct answers to move selection, each with a landmark behind it [6], [10], [14], [3], and they span the two axes that matter: whether the agent searches at decision time, and where its knowledge comes from.

**D3, one shared network.** Giving each paradigm its best-suited architecture would measure engineering effort rather than the paradigms; fixing it trades peak strength for an interpretable result.

**D4, Elo from a full round robin.** Single matches mislead under non-transitivity, established for Xiangqi specifically [17]; fitted ratings with intervals are the standard remedy [20].

**D5, a published corpus.** A scraping attempt was abandoned for a published, citable collection, reproducible by third parties; the cost is no control over sampling, mitigated by replay-validating every game.

@TABLE decisions | Design decisions, rationale, and the cost accepted
| Decision | Choice | Principal reason | Cost accepted |
|---|---|---|---|
| D1 Domain | Xiangqi | Harder than chess, under-studied, open data and engine | Smaller literature to build on |
| D2 Paradigms | Four, spanning search and knowledge source | The four distinct historical answers | Less depth per paradigm |
| D3 Network | One shared policy-value network | Isolates algorithm from architecture | Lower peak strength |
| D4 Metric | Fitted Elo, round robin, 95% CI | Robust to non-transitivity | Many games for tight intervals |
| D5 Data | Published human corpus | Reproducible and citable | No control over sampling |

---

## IV. SOLUTION BLUEPRINT AND RESEARCH DESIGN

### A. The controlled apparatus

The system is a single substrate shared by every agent; only the decision procedure between an encoded position and a chosen action differs. {{f:framework}} lists the components held fixed. Two deserve comment: the legality mask sets illegal logits to negative infinity, so no agent can select an illegal action and no gradient is spent on one; and the single policy-value network is reused by imitation, by Proximal Policy Optimization as actor and critic, and by the search as priors and leaf value, with width and depth configurable for the capacity ablation.

@FIG col | framework | research/figures/paper/p22_apparatus.png | The controlled apparatus. One shared engine, action space and network feed four interchangeable decision functions, the only variable, and every agent is rated on one Elo scale.

### B. The four agents

**M1, Minimax:** alpha-beta negamax with a hand-written material evaluation at depths one to three; it learns nothing and is the control condition. **M2, Imitation learning:** supervised training on human games, with the human's move as policy target and the result as value target. **M3, Self-play reinforcement learning:** action-masked self-play with Proximal Policy Optimization [13], in two variants — M3a from random initialisation and M3b from the M2 checkpoint, whose contrast is the direct test of RQ2. **M4, Tree search with the network:** best-first search using the network's priors for selection and its value head for leaves, playing the most-visited move.

### C. Research design and data

The study is a controlled comparative experiment. The **independent variable** is the paradigm, or in the ablations the single lever varied; the **dependent variable** is fitted Elo, supported by move-prediction accuracy. Everything else is a **control**: the same engine, action space, encoding, architecture and protocol, a matched number of self-play steps where two learning methods are compared, and a fixed seed with the split in a committed manifest. Because only the decision function varies, differences are attributable to the paradigm rather than to the surrounding engineering.

The data are 141,356 legal human games, about 11.6 million positions, split at game level into 113k/14k/14k. Every game is replay-validated through the project's own rule engine, which both cleans the corpus and tests the engine at scale. A separate engine self-play corpus supports the teacher-distillation lever.

---

## V. PROCEDURE: HOW THE OBJECTIVES WILL BE ACHIEVED

**Experiment 1 (RQ1, O3).** A colour-balanced round robin among the four agents plus the random floor, with ratings fitted from the score matrix and reported with 95% confidence intervals, plus a head-to-head matrix. The expected score of a rating difference is

@EQ | elo | <i>E</i><sub>A</sub> = 1 / ( 1 + 10<sup>(<i>R</i><sub>B</sub> &#8722; <i>R</i><sub>A</sub>)/400</sup> )

inverted and fitted over the round robin to obtain each rating, with intervals from the delta method [20].

**Experiment 2 (RQ2, O2).** Imitation-only, imitation-initialised self-play and from-scratch self-play are isolated and the signed Elo differences reported, with learning proxies: top-one move accuracy for imitation, reward and value traces for self-play.

**Experiment 3 (RQ3, O4).** One lever is varied at a time and each agent re-entered into the round robin, so every effect is in the same unit: search simulations from 100 to 800, imitation epochs, teacher (human versus engine distillation), and network size at 1.15M versus 4.85M parameters. A stability study compares fragmented, continuous and buffer-persisted training.

**Metrics.** Fitted and engine-anchored Elo with intervals, head-to-head win rates, the signed Elo difference between the imitation-only and imitation-initialised agents, per-lever effects, top-one accuracy and GPU-hour cost. A difference is called significant only when intervals do not overlap.

**Threats to validity.** Relative Elo depends on the pool, mitigated by an engine anchor and full round robins; few games per pair widen intervals, mitigated by rerunning decisive comparisons; non-transitivity is handled by fitting over complete round robins [17]; and every conclusion is stated as holding at the training budget actually used.

---

## VI. PRELIMINARY RESULTS AND FEASIBILITY

The apparatus is not a plan: a tested implementation exists and a first round of measurements has been taken ({{f:elo}}). The tree-search agent attains the highest rating; imitation learning alone outranks every self-play variant at the budget trained so far; and four attempts to push strength higher — more search, more epochs, a stronger teacher and a four-fold larger network — converged on a similar ceiling, the pattern motivating RQ3. These are interval-wide measurements that the study will formalise with more games; they are reported as evidence of feasibility, not as findings.

@FIG col | elo | research/figures/paper/p13_elo_main.png | Preliminary feasibility measurement: fitted relative Elo with 95% confidence intervals from a round robin of the four paradigms plus a random agent.

---

## VII. EXPECTED CONTRIBUTIONS AND BENEFICIARIES

**Technical.** A reusable, single-interface evaluation league for comparing heterogeneous paradigms on one rule engine, runnable on one consumer GPU; a later student adds a fifth paradigm by implementing one function.

**Research.** The first controlled, Elo-anchored comparison of these four paradigms for Xiangqi; a quantified answer to whether imitation pretraining helps the self-play that follows; and an analysis of what bounds strength.

**Software engineering.** Evidence that training-stability engineering — replay-buffer continuity and best-checkpoint selection — materially affects model quality, giving concrete reproducibility guidance [15].

**Beneficiaries.** Researchers choosing an approach for a new game, who have no like-for-like evidence today; the Xiangqi AI community [17]–[19]; and students here, who inherit a tested environment and a working league.

---

## VIII. SCHEDULE TO THE THESIS DEFENCE

The implementation and preliminary runs exist, so what remains is scale-up, calibration and writing, ending at the defence at the end of May 2027 ({{t:sched}}).

@TABLE sched | Schedule from this proposal to the thesis defence
| Period | Milestone | Output |
|---|---|---|
| Nov 2026 | Finalise proposal; supervisor consultation; freeze the protocol | Approved protocol |
| Dec 2026 | Scale the round robins; add the engine anchor at several depths | Calibrated league |
| Jan 2027 | Scale-up study: larger networks and self-play budgets | Evidence on RQ3 |
| Feb 2027 | Complete the ablations; write Chapters 1 to 3 | Draft chapters 1–3 |
| Mar 2027 | Write Chapter 4, results | Draft chapter 4 |
| Apr 2027 | Write Chapter 5, discussion; full draft to the advisor | Complete draft |
| May 2027 | Revisions; reproducibility check; rehearse and defend | Thesis and defence |

---

## IX. CONCLUSION

This proposal asks a question the literature has left open and a practitioner needs answered: given one game, which of the four principal artificial-intelligence paradigms actually plays it best, and what limits the others? It is answerable because the study fixes the environment, action space, network and rating scale and varies only the decision function, so the result is attributable rather than anecdotal. It is credible because a tested implementation and preliminary measurements already exist. And it is worth doing because its three outputs — a reusable comparison league, the first like-for-like paradigm comparison for Xiangqi, and reproducibility guidance for self-play reinforcement learning — are useful beyond this thesis, and will be reported honestly, including where the evidence is negative.

---

## REFERENCES

[1] D. Silver *et al.*, "Mastering the game of Go with deep neural networks and tree search," *Nature*, vol. 529, no. 7587, pp. 484–489, 2016.

[2] D. Silver *et al.*, "Mastering the game of Go without human knowledge," *Nature*, vol. 550, no. 7676, pp. 354–359, 2017.

[3] D. Silver *et al.*, "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play," *Science*, vol. 362, no. 6419, pp. 1140–1144, 2018.

[4] C. E. Shannon, "Programming a computer for playing chess," *Philosophical Magazine*, vol. 41, no. 314, pp. 256–275, 1950.

[5] D. E. Knuth and R. W. Moore, "An analysis of alpha-beta pruning," *Artificial Intelligence*, vol. 6, no. 4, pp. 293–326, 1975.

[6] M. Campbell, A. J. Hoane Jr., and F.-h. Hsu, "Deep Blue," *Artificial Intelligence*, vol. 134, no. 1–2, pp. 57–83, 2002.

[7] R. Coulom, "Efficient selectivity and backup operators in Monte-Carlo tree search," in *Computers and Games*, LNCS vol. 4630, pp. 72–83, 2006.

[8] L. Kocsis and C. Szepesvári, "Bandit based Monte-Carlo planning," in *Proc. European Conf. Machine Learning (ECML)*, LNCS vol. 4212, pp. 282–293, 2006.

[9] C. B. Browne *et al.*, "A survey of Monte Carlo tree search methods," *IEEE Transactions on Computational Intelligence and AI in Games*, vol. 4, no. 1, pp. 1–43, 2012.

[10] D. A. Pomerleau, "Efficient training of artificial neural networks for autonomous navigation," *Neural Computation*, vol. 3, no. 1, pp. 88–97, 1991.

[11] S. Ross, G. J. Gordon, and J. A. Bagnell, "A reduction of imitation learning and structured prediction to no-regret online learning," in *Proc. 14th Int. Conf. Artificial Intelligence and Statistics (AISTATS)*, PMLR vol. 15, pp. 627–635, 2011.

[12] A. Hussein, M. M. Gaber, E. Elyan, and C. Jayne, "Imitation learning: A survey of learning methods," *ACM Computing Surveys*, vol. 50, no. 2, art. 21, 2017.

[13] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal policy optimization algorithms," arXiv:1707.06347, 2017.

[14] O. Vinyals *et al.*, "Grandmaster level in StarCraft II using multi-agent reinforcement learning," *Nature*, vol. 575, pp. 350–354, 2019.

[15] P. Henderson, R. Islam, P. Bachman, J. Pineau, D. Precup, and D. Meger, "Deep reinforcement learning that matters," in *Proc. AAAI Conf. Artificial Intelligence*, vol. 32, no. 1, 2018.

[16] Y. Tian *et al.*, "ELF OpenGo: An analysis and open reimplementation of AlphaZero," in *Proc. 36th Int. Conf. Machine Learning (ICML)*, PMLR vol. 97, pp. 6244–6253, 2019.

[17] Y. Li *et al.*, "Mastering Xiangqi by tackling non-transitivity in two-player zero-sum games," *Transactions on Machine Learning Research (TMLR)*, 2023, arXiv:2308.04719.

[18] B. Yilmaz, J. Hu, and J. Liu, "Deep reinforcement learning Xiangqi player with Monte Carlo tree search," arXiv:2506.15880, 2025.

[19] Y. Chen, S. Liu, Y. Lyu, C. Zhang, J. Shi, and T. Xu, "Xiangqi-R1: Enhancing spatial strategic reasoning in LLMs for Chinese chess via reinforcement learning," arXiv:2507.12215, 2025.

[20] A. E. Elo, *The Rating of Chessplayers, Past and Present*. New York: Arco, 1978.

