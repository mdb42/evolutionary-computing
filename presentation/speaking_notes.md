# Presentation Speaking Notes

## Slide 0 - Opening Slide

A Novel Pareto-optimal Ranking Method for Comparing Multi-objective Optimization Algorithms

by Ibrahim, Bidgoli, Rahnamayan, and Deb.

## Slide 1 - Outline

We'll start with a brief introduction describing the motivation and core challenge.
We'll then review some background on multi-objective optimization and performance indicators.
Following that, I'll present the proposed Pareto-optimal ranking method in detail, 
then walk through experimental validation using real competition data.
Finally, we'll conclude with the paper's key findings and broader implications.

## Slide 2 - Introduction

When we create new multi-objective algorithms, we face a tricky question... which one's best?

Unlike single-objective problems with one clear winner, these algorithms produce entire Pareto fronts of trade-off solutions

Metrics proliferation... Research community has developed dozens of metrics to measure performance

Some check how close you get to the optimal solution
Others look at how well-spread your solutions are

The paper's approach... Uses 10 of these metrics

Core problem... They often disagree

One algorithm might do great on hypervolume test but fail miserably at spacing

Current methods limitation... Just compute all scores separately and somehow mash them together, often with arbitrary weights

Paper's solution... Treat the metrics themselves as objectives and use Pareto optimality to rank them fairly

## Slide 3 - Background Review

Left side - Standard multi-objective optimization problem...

Trying to optimize M different objectives at once
Example: cost, performance, and reliability all competing for attention
Each decision variable has to stay within its bounds
Might have additional constraints to satisfy

Right side - The dominance concept... MAGIC! Hocus pocus! Abracadabra!

$$
\forall i \in \{1, 2, \ldots, M\}, \; f_i(x) \leq f_i(\hat{x})
$$

$$
\exists j \in \{1, 2, \ldots, M\}, \; f_j(x) < f_j(\hat{x})
$$

For solution x to dominate solution x-hat...
Needs to be at least as good in every objective
AND strictly better in at least one

Solutions that nobody dominates will form the Pareto front

Non-Dominated Sorting algorithm...
Sorts all solutions into levels
Level 1: non-dominated solutions
Level 2: solutions only dominated by level 1
And so on...

Key insight...
Paper takes this exact ranking system, originally meant for solutions, and applies it to rank algorithms based on their performance metrics

## Slide 4 - Hypervolume (HV) Indicator

$$
HV(A) = vol\left( \bigcup_{a \in A} [f_1(a), r_1] \times [f_2(a), r_2] \times \ldots \times [f_M(a), r_M] \right)
$$

Most popular metric... Hypervolume

Measures volume of space dominated by your Pareto front compared to a reference point
Imagine drawing boxes from each solution to a reference point and calculating total volume
Bigger volume means better performance

Strengths...
Captures both how close you are to optimal front
AND how well-spread your solutions are

Weakness... Gets computationally expensive with many objectives

Not for me though! I have a 5060. Eat it, losers.

## Slide 5 - Generational Distance (GD)

$$
GD(S, P) = \frac{\sqrt{\sum_{i=1}^{|S|} dist(i, P)^2}}{|S|}
$$

Core question... How close did we get?

Measurement... Average distance from your solutions to the true Pareto front

Calculation method...
Distance from each solution to its nearest neighbor on true front
Take root mean square
Smaller values are better

Major limitation... Only cares about convergence
Could cluster all solutions in one spot near the front
Still get a great score

## Slide 6 - Inverted Generational Distance (IGD)

$$
IGD(S, P) = \frac{\sqrt{\sum_{i=1}^{|P|} dist(i, S)^2}}{|P|}
$$

Flip: Instead of measuring from your solutions to true front, measures from true front to your solutions
IGD captures both convergence AND diversity

Example of diversity capture...
If you cluster solutions in one area
Points on true front far from your cluster have large distances
Hurts your score

Competition standard... One of two metrics used in CEC competition
Tells a more complete story than GD alone

## Slide 7 - Two-set Coverage (C)

$$
C(A, B) = \frac{|\{b \in B, \text{ there exists } a \in A \text{ such that } a \succ b\}|}{|B|}
$$

Unique aspect... Only metric that directly compares two algorithms head-to-head

C(A,B) meaning... What fraction of algorithm B's solutions are dominated by algorithm A

If 0.25, A dominates 25% of B's solutions

Key characteristic: Not symmetric
A might dominate 25% of B's solutions
While B dominates 60% of A's
Need to check both directions

Gives competitive insights you can't get from other metrics
Shows not just how good an algorithm is
But how it stacks up against specific rivals

## Slide 8 - Coverage over the Pareto front (CPF)

$$
P' = \{argmin_{x \in R} \|f(x) - r\| \mid p \in P\}
$$
$$
CPF = \frac{Vol(P')}{Vol(R)}
$$

Diversity through projection... Clever projection trick

Process:
Takes your solutions
Projects them down one dimension
Maps each to nearest reference point on true Pareto front
Mathematical gymnastics involving volume calculations

Result: Number between 0 and 1
0.8 means solutions cover 80% of reference space

Catch: Results depend heavily on reference points chosen
Different reference sets give different coverage scores for same solutions

Still useful: Good way to see how well solutions span entire front

## Slide 9 - Hausdorff Distance to the Pareto Front

$$
\Delta_p(S, P) = \max(GD(S, P), IGD(S, P))
$$

Simple approach: Maximum of GD and IGD

Worst-case metric that exposes any weakness

Great convergence with GD of 0.1
Terrible coverage with IGD of 0.8
Hausdorff reports 0.8, highlighting the problem
    
Can't game it... Need to be good at both convergence and diversity

Perfect for when you need algorithms that reliably deliver both accurate and well-spread solutions

## Slide 10 - Pure Diversity (PD)

$$
PD(A) = \max_{s_i \in A} (PD(A - s_i) + d(s_i, A - s_i))
$$

$$
d(s, A) = \min_{s_j \in A} (dissimilarity(s, s_j))
$$

Does what it says... Measures spread while completely ignoring solution quality

Algorithm: 
Recursively picks solution most different from all others
Builds up total diversity score
Higher values mean better spread

Use case...
Exploring unknown territory
Early optimization stages where you just want to cover ground

Obvious downside: Beautifully diverse solutions might be nowhere near actual Pareto front

Always paired with convergence metrics
diversity alone tells only half the story

## Slide 11 - Spacing (SP)

$$
SP(S) = \frac{1}{|S| - 1} \sum_{i=1}^{|S|} (d - d_i)^2
$$

Analogy: Think fence posts versus random clusters

Measures: How evenly solutions are distributed

Calculation... Standard deviation of distances between neighboring solutions
Lower values are better
Zero means perfect uniformity

Detection capability... Catches bunching in some areas and gaps in others

Limitation... Blind to quality
Could have beautifully spaced solutions that are all terrible
Usage: Nobody uses spacing alone; always part of bigger picture with convergence metrics

## Slide 12 - Overall Pareto Spread (OS)

$$
OS(S) = \prod_{i=1}^{m} \left( \frac{\max_{x \in S} f_i(x) - \min_{x \in S} f_i(x)}{f_i(P_G) - f_i(P_B)} \right)
$$

Focus: How much of possible objective space your solutions cover

Per-objective calculation...
What percentage of theoretical range achieved
Example: Solutions span 2 to 8 on objective that could range 0 to 10 = 60% coverage

Overall score... Multiplies percentages across all objectives
Need good spread in everything to score well

Weakness: Few extreme solutions at boundaries can fool it
Might score great with just corner solutions
While having huge gaps in between

## Slide 13 - Distribution Metric (DM)

$$
DM(S) = \frac{1}{|S|} \sum_{i=1}^{m} \left( \frac{\sigma_i}{\mu_i} \cdot \frac{|f_i(P_G) - f_i(P_B)|}{R_i} \right)
$$

Approach... Gets fancy by combining spread and uniformity measures

Per-objective process:
Calculates coefficient of variation (how spread out relative to average)
Normalizes across different objective scales

Lower values mean better distribution that's both well-spread AND uniform

But wait!...
Uses coefficient of variation (σ/μ) and ratios of ranges
Can be overly sensitive to extreme values
Single outlier could drastically affect both standard deviation and achieved range
Potentially giving misleading values

## Slide 13.5???

Review all the metrics together...

## Slide 14 - Proposed Ranking Method - Steps

With all the metrics covered, we've seen the challenge: each metric captures something different about convergence, diversity, or distribution - but how do we combine them fairly?

Solution introduction: The authors propose a Pareto-optimal ranking method

Let's walk through this step by step...

Step 1: Select M performance indicators
In their experiments, authors used all 10 metrics we just reviewed

Step 2: Run each algorithm R times
They used 20 runs

Step 3: For each algorithm and each run, calculate ALL M metric scores
This gives us an R-by-M matrix per algorithm

Step 4: Concatenate everything into one massive dataset
With A algorithms, R runs each, and M metrics
(slow down)
Get A × R × M-dimensional points
Each point represents one run of one algorithm evaluated on all metrics

Step 5 - Key insight: Treat these metric scores AS IF they were objectives in a multi-objective problem
Just like we use non-dominated sorting on solutions, apply it to algorithm performances

Important: Need to standardize directions
Some metrics like HV are: bigger is better
Others like IGD are: smaller is better
Authors flip maximization metrics (inverse or negate)
Everything becomes minimization problem

Step 6: Count how many times each algorithm appears in each Pareto level
Algorithms dominating others across multiple metrics cluster in Level 1
Weaker performers sink to lower levels

But how exactly do we convert these level assignments into final rankings?
That's where our four ranking techniques come in...

## Slide 15 - Olympic Method

The Olympic method is dead simple
whoever has the most points in level 1 wins, just like counting gold medals

Example:
Algorithm a1 has 20 points in level 1
Algorithm a2 has 15 points in level 1
So a1 takes first place

Tie-breaking: Ties get broken by level 2, then level 3, and so on

The catch: Ignores overall consistency
Notice a2 actually performs better in level 2 (14 points versus 10)
But Olympic doesn't care

It's all about peak performance

When to use: Perfect when you need the absolute best, even if it's occasionally inconsistent

## Slide 16 - Linear Method

$$
Adaptive\_Score(a_i) = \sum_{l=1}^{L} \frac{CW(a_i, l)}{Total\_CW(l)}
\tag{21}
$$

Improvement attempt... Linear method attempts to improve upon Olympic's tunnel vision

How it works: Counting all levels with decreasing weights
Level 1 gets weight 3
Level 2 gets weight 2
Level 3 gets weight 1

Example calculation...
a1 scores: 20×3 + 10×2 + 1×1 = 81
a2 scores: 15×3 + 14×2 + 2×1 = 75

Result comparison...
a1 still wins but by much smaller margin
81 to 75 instead of 20 to 15
Because a2's strong level 2 performance now matters

Philosophy... Rewards algorithms that perform well across the board while still prioritizing excellence

## Slide 17 - Exponential Method

Splits the difference between Olympic and Linear
Uses weights that halve at each level-1, 0.5, 0.25, and so on

Example calculation:
a1 scores: 20×1 + 10×0.5 + 1×0.25 = 25.25
a2 scores: 15×1 + 14×0.5 + 2×0.25 = 22.5

Result analysis...
a1 wins more decisively
Lower levels barely matter
a2's level 2 advantage gets heavily discounted
    
Creates middle ground:
Still consider all levels unlike Olympic
But excellence at top is exponentially more valuable than steady mediocrity

When to use: Great when you want algorithms that usually excel but won't completely tank on bad days

## Slide 18 - Adaptive Method

The most sophisticated method, using cumulative weights and proportions

Step 1: Calculates cumulative points
Level 2 includes all points from levels 1 AND 2

Core question... What fraction of the total does each algorithm contribute at each level?

Example...
a1's cumulative weights: 20, 30, 31
a2's cumulative weights: 15, 29, 31
Computing proportions and summing:
a1 scores 1.58
a2 scores 1.42

Key advantage: Adapts to whatever distribution of algorithms you throw at it
No arbitrary weights needed

Authors' recommendation...
Captures both absolute performance and relative dominance
Essentially asks 'what share of total quality does each algorithm provide?'

Perfect for robust rankings across different scenarios

## Slide 19 - Experimental Validation: Experimental Settings (removal candidate)

Data source: The experimental validation uses data from the 2018 CEC many-objective optimization competition

10 algorithms tested
15 MaF benchmark problems

Problem diversity... These problems cover everything from simple linear functions to mixed, disconnected, multimodal problems

Experimental setup...
Each algorithm ran 20 times
On problems with 3, 5, 10, and 15 objectives

Key difference: Original competition only used IGD and HV to rank algorithms
This study uses all 10 metrics we discussed earlier

## Slide 20 - Ranking algorithms when solving one specific test problem with a particular number of objectives

What tables show... How each algorithm's 20 runs spread across Pareto levels

Example 1 - 5-objective MaF1:
AGE-II dominates with 17 runs in level 1
RVEA scatters across 7 levels-totally inconsistent

Example 2 - 15-objective MaF10:
CVEA3 and HhMOEA achieve perfect scores
All 20 runs in level 1

RadViz visualization:
Each point is one run
Each axis is a metric
Can see the separation between Pareto levels

Ranking Quirks...
AMPDEA ranks 10th by Olympic but 9th by other methods
Has fewer top-tier runs but better overall performance

Key insight... Shows how algorithm behavior changes drastically between different problems

## Slide 21 - Ranking algorithms when solving a set of test problems with a particular number of objectives

Now we aggregate across all 15 problems
Each row sums to 300 (20 runs × 15 problems)

10-objective results: CVEA3 dominates with 274 level-1 points

15-objective shift: fastCAR takes over with 238

Difficulty spike evidence...
15 objectives: algorithms spread across 12 Pareto levels
10 objectives: just 8 levels

RadViz patterns...
10 objectives: concentric rings
15 objectives: spiral pattern as algorithms struggle with complexity??

Interesting case - HhcMOEA...
Ranks 2nd by Olympic
Drops to 6th by Linear method
Why? Produces some brilliant runs but lacks consistency
Exactly the nuance that single metrics miss

## Slide 22 - Determining the overall rankings of algorithms

Total data: This table shows all 900 data points across 18 Pareto levels

Top performers:
FastCAR and HhcMOEA both dominate with 747 level-1 points
That's 83% of their runs

Key divergence:
fastCAR stays consistent through level 9
HhcMOEA spreads to level 11

Extreme cases...
RVEA: only algorithm hitting all 18 levels-high variability
RPEA: fewest level-1 points but solid middle-tier presence

Distribution impact... This distribution drives our final rankings

## Slide 23 - Determining the overall rankings of algorithms

Clear winner: FastCAR takes unanimous first place across all four methods-true dominance

Consistent performers:
RPEA consistently ranks last
RVEA consistently 9th

Middle tier variation:
AGE-II ranges from 5th to 7th depending on method
AMPDEA inversely goes from 7th to 5th

Average ranks: Rightmost column smooths out these differences

## Slide 24 - Comparison of rankings from the Competition and the proposed method

Moment of truth: Comparing our results to official competition that used only IGD and HV

When restricted to two metrics...
Get similar results
CVEA3 first, AMPDEA second, RPEA last

Key discrepancies...
KnEA ranked 4th in competition
But 6th-7th in our methods
Apparently games IGD and HV specifically

The lesson... Two metrics miss crucial performance aspects that our 10-metric approach captures

## Slide 25 - Conclusion

In conclusion, this paper solves a fundamental problem: how to fairly compare algorithms using multiple conflicting metrics

Key insight: Treat the metrics themselves as objectives and use Pareto optimality to rank them

Method advantages:
Parameter-free
Scalable
Offers four ranking methods from Olympic's focus to Adaptive's balanced approach

Broader applications... This framework can extend to any multi-criteria decision:
Machine learning
Healthcare
Business

And honestly, I found it super interesting. Using multi-objective methods to evaluate multi-objective algorithms

Very meta!

Hey, I heard you like optimizers, so I put optimizers in your optimizers.
