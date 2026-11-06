# Benchmarking Minimax, Imitation, Self-Play Reinforcement Learning, and Monte Carlo Tree Search in a Multi-Agent Xiangqi (象棋) League

@META Alma Nurul Salma (诺艾玛)
@META Nanjing Xiaozhuang University (南京晓庄学院) · Student ID: L25020007
@META almanurulsalma@gmail.com · Research proposal · 6 November 2026

@ABSTRACT
Four distinct families of artificial intelligence have each produced a landmark result in board games: adversarial tree search, supervised imitation of human experts, self-play reinforcement learning, and Monte Carlo tree search guided by a learned network. They have never been compared with one another. Each published demonstration uses a different game, a different architecture and a different amount of computation, so a practitioner cannot tell which approach is actually the strongest, nor what limits the others. This proposal sets out a controlled study that removes those confounds. All four paradigms are implemented on one domain, Xiangqi (Chinese chess), over a single shared rule engine, a single 2,086-move action space, a single policy-value network and a single Elo league, so that the only variable is the learning algorithm. The study ranks the paradigms with confidence intervals, measures whether imitation pretraining helps or hinders the self-play that follows, and runs a four-lever ablation to identify what actually bounds playing strength. A complete and tested implementation with preliminary measurements already exists, so the work is a rigorous formalisation rather than a speculative build. The expected contributions are a reusable multi-agent evaluation league, the first like-for-like paradigm comparison for Xiangqi, and reproducibility guidance for self-play reinforcement learning.

@KEYWORDS Xiangqi, Chinese chess, game artificial intelligence, Monte Carlo tree search, imitation learning, Proximal Policy Optimization, Elo rating, reproducibility

---

## I. INTRODUCTION

### A. Background

The dominant recipe for superhuman board-game play was established on Go by AlphaGo, which combined supervised imitation of expert moves, reinforcement learning by self-play, and Monte Carlo tree search [1]. AlphaGo Zero removed the human-data stage and learned tabula rasa [2], and AlphaZero generalised the same algorithm to chess, shogi and Go [3]. Each of the components those systems combine has its own independent lineage: adversarial search was formalised for chess by Shannon [4] and made practical by the alpha-beta analysis of Knuth and Moore [5], reaching the human world champion in Deep Blue [6]; behaviour cloning originates in training a network to imitate a human driver [10]; and the tree search itself was introduced by Coulom [7] and given its bandit-based selection rule by Kocsis and Szepesvári [8].

@FIG col | board | research/proposal/figures/gfig1_board.png | The 9 x 10 Xiangqi board with its river and two palaces, shown in the opening position. Red moves first.

@FIG col | moves | research/proposal/figures/gfig2_moves.png | How each piece moves and captures. The jump-the-screen cannon capture and the blockable horse leg have no chess analogue and enlarge the branching factor the agents must search.

### B. The problem, and where it comes from

**Problem.** The four paradigms are never compared against one another. They are compared against humans, or against the previous state of the art in whichever domain the authors chose. Consequently nobody can say which of them plays a given game best, or what limits each of them, because no two published results share a domain, an architecture and a training budget.

This is not a suspicion of the author's; it is a documented methodological failure in the field. Henderson *et al.* showed that reported differences between deep reinforcement-learning algorithms are frequently attributable to implementation details, random seeds and evaluation choices rather than to the algorithms being compared, and they argued explicitly for controlled comparisons in a shared codebase with reported variance [15]. The problem therefore originates in the published literature itself, and the present project is a direct response to that recommendation applied to a domain in which it has not yet been attempted.

The domain chosen is Xiangqi (象棋, Chinese chess), shown in Figures 1 and 2: a culturally major but academically under-served game with a 9 x 10 board, a higher branching factor than chess, a cannon that captures by jumping a screen, a horse that can be blocked, and the unusual rule that stalemate is a loss. Reported figures place its branching factor near thirty-eight against roughly thirty-five for chess and its game-tree complexity near 10^150 against roughly 10^123 [18].

**Solution.** Hold the environment, the state representation, the action space, the network and the evaluation protocol fixed, and swap only the learning paradigm. A like-for-like ranking then isolates the effect of the paradigm itself, and targeted ablations attribute the remaining differences to search, capacity, data or algorithm.

### C. Research questions

- **RQ1 (ranking).** With the domain, representation, architecture and evaluation protocol held constant, how do the four paradigms rank on a single Elo scale?
- **RQ2 (effect of imitation pretraining).** Does initialising self-play reinforcement learning from an imitation-learned policy improve the final agent, or constrain it? The literature motivates both outcomes: imitation supplies a strong starting policy [12], but it also imposes the demonstrator's state distribution, which becomes a liability once the policy departs from it [11].
- **RQ3 (binding constraint).** When an agent stops improving, which factor is responsible: play-time search, model capacity, the quantity or quality of the training data, or the learning algorithm?

### D. Objectives

- **O1.** Implement a rules-correct Xiangqi environment with a fixed action space and a tensor encoding shared by every learned agent.
- **O2.** Implement the four paradigms over that shared substrate, together with a random floor and an engine reference.
- **O3.** Build an evaluation league reporting fitted Elo with 95% confidence intervals and a held-out move-prediction metric.
- **O4.** Run ablations that isolate search budget, network capacity, teacher quality and training length, to answer RQ3.
- **O5.** Release the environment, the datasets, the checkpoints and a demonstrator so that the comparison can be rerun and extended.

---

## II. RELATED WORK AND THE RESEARCH GAP

Each source below is identified with the specific point it contributes to this study.

**Adversarial search.** Shannon formalised minimax for chess [4]; Knuth and Moore quantified the savings from alpha-beta pruning [5]; Campbell *et al.* documented Deep Blue [6]. *Useful point:* this paradigm requires no data and no training, so it supplies a learning-free control condition against which every learned agent can be measured.

**Imitation learning.** Pomerleau demonstrated behaviour cloning on a driving task [10]; Hussein *et al.* survey the methods and their failure modes [12]; Ross *et al.* formalised the compounding-error problem caused by distribution shift [11]. *Useful point:* theory predicts that an imitator is strong where the demonstrations are dense and degrades where they thin out, which gives RQ2 a falsifiable prediction rather than a guess.

**Self-play reinforcement learning.** Sutton and Barto give the formal framework [21]; Schulman *et al.* introduced Proximal Policy Optimization, chosen here for its stability [13]; Vinyals *et al.* reached grandmaster level by self-play in a far less structured domain [14]. *Useful point:* self-play needs no human data but is sample-hungry, which makes it the paradigm most sensitive to a limited training budget and therefore the most important one to measure honestly.

**Monte Carlo tree search with a learned network.** Coulom introduced the search [7], Kocsis and Szepesvári supplied the selection rule [8], and Browne *et al.* survey the design space [9]; the AlphaGo family combined the search with deep networks [1]–[3]. *Useful point:* this is the strongest known recipe, and its selection rule is the one the search agent in this study implements.

**Reproducibility.** Henderson *et al.* demonstrated that policy-gradient results vary sharply with seeds and implementation [15], and Tian *et al.* reimplemented AlphaZero openly and reported the resources it required [16]. *Useful point:* together these justify the interval-based evaluation, the shared codebase, and the explicit reporting of the training budget used here.

**Xiangqi-specific work.** Li *et al.* introduced JiangJun, which combines tree search with population methods over more than ten thousand human games to address strategic non-transitivity [17]. Yilmaz *et al.* describe an AlphaZero-style Xiangqi player and report the complexity figures used above [18]. Chen *et al.* use Xiangqi as a testbed for spatial reasoning in language models trained with policy optimization [19]. *Useful point:* these three establish Xiangqi as a venue for serious machine-learning research, which removes any doubt about the domain's legitimacy; each, however, optimises a single pipeline and none compares paradigms.

**Research gap.** Table I makes the position explicit. No controlled, Elo-anchored comparison of classical search, self-play reinforcement learning, imitation-initialised reinforcement learning, and AlphaZero-style tree search exists for Xiangqi. Consequently both the direction of the imitation-pretraining effect and the binding constraint on strength remain unquantified for this game. This is primarily an evaluation and methodology gap, with secondary performance and application dimensions.

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

Five decisions determine the shape of the study. Each is recorded with the reason it was taken and the cost it carries, so that the design can be challenged on its merits (Table II).

**D1, the domain.** Xiangqi was chosen for three reasons. Scientifically, it is demonstrably harder to search than chess [18] while remaining a perfect-information, zero-sum, two-player game, so the classical theory applies unmodified and the comparison stays clean. In terms of the gap, chess and Go are saturated benchmarks where a further comparison adds little, whereas Xiangqi has very few published studies and none that compare paradigms. Practically, a large corpus of recorded human games is publicly available and a strong open-source engine exists as a reference, which makes rigorous evaluation possible without proprietary resources.

**D2, the four paradigms.** These four are not an arbitrary selection: they are the four historically distinct answers to the move-selection problem, each with a landmark result behind it [6], [10], [14], [3]. They also span the two axes that matter — whether the agent searches at decision time, and whether its knowledge is hand-written, learned from humans, or self-generated. A fifth method would occupy a cell that is already filled.

**D3, one shared network.** If each paradigm were given the architecture that suits it best, the study would measure the author's engineering effort per paradigm rather than the paradigms themselves. Fixing the architecture sacrifices some peak strength in exchange for an interpretable result, which is the stated aim.

**D4, Elo from a full round robin.** Single matches are misleading in a domain with non-transitivity, a property established for Xiangqi specifically [17]. A colour-balanced round robin with fitted ratings and confidence intervals is the standard remedy and makes the ranking defensible [20].

**D5, a published corpus rather than a self-built scrape.** An attempt to scrape games directly was abandoned in favour of a published, citable collection. A published corpus is reproducible by a third party and removes legal ambiguity; the cost is that the author does not control the sampling, which is mitigated by replay-validating every game against the rule engine.

@TABLE decisions | Design decisions, rationale, and the cost accepted
| Decision | Choice | Principal reason | Cost accepted |
|---|---|---|---|
| D1 Domain | Xiangqi | Harder than chess, under-studied, open data and engine available | Smaller existing literature to build on |
| D2 Paradigms | Four, spanning search and knowledge source | They are the four distinct historical answers | Less depth per paradigm than a single-method thesis |
| D3 Network | One shared policy-value network | Isolates the algorithm from the architecture | Lower peak strength per paradigm |
| D4 Metric | Fitted Elo, full round robin, 95% CI | Robust to non-transitivity; defensible ranking | Many games needed for tight intervals |
| D5 Data | Published human corpus | Reproducible and citable | No control over sampling |

---

## IV. SOLUTION BLUEPRINT AND RESEARCH DESIGN

### A. The controlled apparatus

The system is a single substrate shared by every agent; only the decision procedure between an encoded position and a chosen action differs between paradigms (Figure 3). Five components are held fixed. A tested rule engine exposed through a standard reinforcement-learning interface generates legal moves. A fixed action space of 2,086 entries indexes every geometrically possible move, with a legality mask that sets illegal logits to negative infinity so that no agent can ever select an illegal action. A 14 x 10 x 9 side-to-move-relative encoding represents the board (Figure 4). One AlphaZero-style policy-value network is reused by imitation, by Proximal Policy Optimization as actor and critic, and by the tree search as priors and leaf value; its width and depth are configurable for the capacity ablation. One Elo league rates every agent.

@FIG span | framework | research/proposal/figures/gfig4_framework.png | The controlled apparatus. One shared engine, action space and policy-value network feed four interchangeable decision functions, which are the only variable, and all agents are rated on a single Elo scale by a colour-balanced round robin anchored to an open-source engine.

@FIG col | tensor | research/proposal/figures/gfig3_tensor.png | The board is encoded as a 14 x 10 x 9 tensor, one 10 x 9 binary plane per piece type and side, always from the perspective of the side to move, and shared by every agent.

### B. The four agents

**M1, Minimax.** Alpha-beta negamax search with a hand-written material evaluation, at depths one to three. It learns nothing, and is therefore the control condition.

**M2, Imitation learning.** Supervised training on recorded human games: the policy target is the move the human played and the value target is the eventual result.

**M3, Self-play reinforcement learning.** Action-masked self-play optimised with Proximal Policy Optimization [13], trained in two variants — M3a from random initialisation and M3b initialised from the M2 checkpoint. The contrast between them is the direct test of RQ2.

**M4, Monte Carlo tree search with the network.** A best-first search that uses the network's priors to guide selection and its value head to evaluate leaves, playing the most-visited move. A random agent is included as a floor, and an open-source engine as a reference point.

### C. Research design

The study is a controlled comparative experiment. The **independent variable** is the paradigm, or in the ablations the single lever being varied. The **dependent variable** is fitted Elo, supported by move-prediction accuracy as a secondary measure of what each network has learned. Everything else is a **control**: the same rule engine, the same action space, the same encoding, the same architecture, the same evaluation protocol, a matched number of self-play steps where two learning methods are compared, and a fixed seed with the data split recorded in a committed manifest. Because only the decision function varies, differences in strength are attributable to the paradigm rather than to the surrounding engineering.

### D. Data

141,356 legal human games, approximately 11.6 million positions, are split at the game level into 113k/14k/14k training, validation and test sets and recorded in a committed manifest for reproducibility. Every game is replay-validated through the project's own rule engine before use, which both cleans the corpus and serves as a large-scale conformance test of the engine. A separate engine self-play corpus supports the teacher-distillation lever of RQ3.

---

## V. PROCEDURE: HOW THE OBJECTIVES WILL BE ACHIEVED

Each objective is discharged by a specific experiment, so that the link between question, procedure and evidence is explicit.

**Experiment 1 (RQ1, O3).** A colour-balanced round robin among the four agents plus the random floor, with ratings fitted from the score matrix and reported with 95% confidence intervals, together with a head-to-head win-rate matrix. The expected score of a rating difference is

@EQ | elo | <i>E</i><sub>A</sub> = 1 / ( 1 + 10<sup>(<i>R</i><sub>B</sub> &#8722; <i>R</i><sub>A</sub>)/400</sup> )

which is inverted and fitted over the round-robin results to obtain each rating, with intervals from the delta method [20].

**Experiment 2 (RQ2, O2).** Imitation-only, imitation-initialised self-play, and from-scratch self-play are isolated and the signed Elo differences reported, alongside learning proxies: top-one move accuracy for the imitation stage, and reward and value traces for self-play.

**Experiment 3 (RQ3, O4).** One lever is varied at a time and each resulting agent is re-entered into the round robin, so that every lever's effect is expressed in the same unit. The levers are search simulations from 100 to 800, imitation epochs, teacher (human games versus engine distillation), and network size at 1.15M versus 4.85M parameters. A separate stability study compares fragmented training, continuous training, and training with a persisted replay buffer.

**Evaluation metrics.** Fitted and engine-anchored Elo with 95% confidence intervals, head-to-head win rates, the signed Elo difference between the imitation-only and imitation-initialised agents, per-lever Elo effects, top-one move accuracy, value-loss trajectories, and the training cost in GPU-hours. A difference is called significant only when the confidence intervals do not overlap.

**Threats to validity.** Relative Elo depends on the pool, which is mitigated by an absolute engine anchor and by full round robins. Few games per pair widen the intervals, which is mitigated by rerunning decisive comparisons with more games. Non-transitivity is handled by fitting ratings over complete round robins rather than by single matches [17]. The shared-substrate design preserves internal validity because only the decision function varies. Finally, all conclusions are stated as holding at the training budget actually used, which is reported beside every result rather than left implicit.

---

## VI. PRELIMINARY RESULTS AND FEASIBILITY

The apparatus is not a plan: a complete, tested implementation already exists, and a first round of measurements has been taken. Figure 5 shows fitted relative Elo from a round robin of the four paradigms plus the random agent. Three observations are already visible. The tree-search agent attains the highest rating. Imitation learning alone outranks every self-play variant at the budget trained so far. And four separate attempts to push strength higher — more search, more epochs, a stronger teacher, and a four-fold larger network — all converged on a similar ceiling, which is the pattern that motivates RQ3. These are preliminary, interval-wide measurements that the proposed study will formalise with more games and tighter intervals; they are reported here as evidence of feasibility, not as findings.

@FIG col | elo | research/figures/paper/p13_elo_main.png | Preliminary feasibility measurement: fitted relative Elo with 95% confidence intervals from a round robin of the four paradigms plus a random agent. A stabilised rerun reproduced the same top rank.

---

## VII. EXPECTED CONTRIBUTIONS AND BENEFICIARIES

**Technical.** A reusable, single-interface multi-agent evaluation league for comparing heterogeneous game-playing paradigms on one rule engine, runnable on a single consumer GPU. A subsequent student can add a fifth paradigm by implementing one function.

**Research.** The first controlled, Elo-anchored comparison of these four paradigms for Xiangqi; a quantified answer to whether imitation pretraining helps the self-play that follows; and a multi-lever analysis of what actually bounds strength.

**Software engineering.** Evidence that training-stability engineering — replay-buffer continuity and best-checkpoint selection — materially affects the quality of the learned model, yielding concrete reproducibility guidance for self-play reinforcement learning, in the spirit of the recommendations of Henderson *et al.* [15].

**Beneficiaries.** Researchers choosing an approach for a new game, who currently have no like-for-like evidence; the small Xiangqi AI community [17]–[19]; and students at this university, who inherit a tested environment, three documented datasets and a working league.

---

## VIII. SCHEDULE TO THE THESIS DEFENCE

The implementation and the preliminary runs already exist, so the remaining work is scale-up, calibration and writing. Table III gives the schedule and Figure 6 plots it against the calendar; the defence is at the end of May 2027.

@TABLE sched | Schedule from this proposal to the thesis defence
| Period | Milestone | Output |
|---|---|---|
| Nov 2026 | Finalise proposal; supervisor consultation; freeze the experimental protocol | Approved protocol |
| Dec 2026 | Scale the round robins for tighter intervals; add the engine anchor at several depths | Calibrated league |
| Jan 2027 | Scale-up study: larger networks and larger self-play budgets | Evidence on RQ3 |
| Feb 2027 | Complete the ablations; write Chapters 1 to 3 | Draft chapters 1 to 3 |
| Mar 2027 | Write Chapter 4, results | Draft chapter 4 |
| Apr 2027 | Write Chapter 5, discussion; full draft to the advisor | Complete draft |
| May 2027 | Revisions; reproducibility check; rehearse and defend | Thesis and defence |

@FIG span | timeline | research/figures/paper/p21_timeline.png | The schedule plotted against the calendar. Solid bars are work already completed; hatched bars are the work remaining to the defence at the end of May 2027.

---

## IX. CONCLUSION

This proposal asks a question that the literature has left open and that a practitioner genuinely needs answered: given one game, which of the four principal artificial-intelligence paradigms actually plays it best, and what limits the others? The question is answerable because the study holds the environment, the action space, the network and the evaluation scale fixed and varies only the decision function, so the result is attributable rather than anecdotal. It is credible because a complete, tested implementation and preliminary measurements already exist, which converts the remaining work from a speculative build into a rigorous formalisation. And it is worth doing because its three outputs — a reusable comparison league, the first like-for-like paradigm comparison for Xiangqi, and concrete reproducibility guidance for self-play reinforcement learning — are useful beyond this thesis, and are reported honestly, including where the evidence turns out to be negative.

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

[21] R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed. Cambridge, MA: MIT Press, 2018.
