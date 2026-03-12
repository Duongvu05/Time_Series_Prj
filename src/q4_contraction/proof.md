# Q4: Contraction Mapping Theorem – Formal Proof

**Source:** Silver Lecture 2 (Bellman equations); proof based on Banach Fixed-Point Theorem.

---

## 1. Setup and Notation

Let $(S, \mathcal{A}, P, R, \gamma)$ be a finite MDP with discount factor $\gamma \in [0, 1)$.

The space of bounded value functions:

$$\mathcal{V} = \{V : S \to \mathbb{R} \mid V \text{ bounded}\}$$

is a **complete metric space** under the sup-norm:

$$\|V\|_\infty = \max_{s \in S} |V(s)|, \quad d(V_1, V_2) = \|V_1 - V_2\|_\infty$$

---

## 2. The Contraction Mapping Theorem (Banach Fixed-Point Theorem)

**Theorem.** Let $(X, d)$ be a complete metric space. If $T : X \to X$ is a **$\gamma$-contraction**, i.e.:

$$d(Tx, Ty) \leq \gamma \cdot d(x, y) \quad \forall x, y \in X, \quad \gamma \in [0,1)$$

then:
1. $T$ has a **unique fixed point** $x^*$ (i.e., $Tx^* = x^*$).
2. The iterates $x_{k+1} = Tx_k$ converge to $x^*$ for **any** starting point $x_0$.
3. The convergence rate satisfies:

$$d(x_k, x^*) \leq \gamma^k \cdot d(x_0, x^*)$$

---

## 3. Bellman Expectation Operator is a Contraction

Define the **Bellman Expectation Operator** $T^\pi : \mathcal{V} \to \mathcal{V}$:

$$(T^\pi V)(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)\bigl[R(s,a,s') + \gamma\,V(s')\bigr]$$

**Claim:** $T^\pi$ is a $\gamma$-contraction under $\|\cdot\|_\infty$.

**Proof:**

Let $V_1, V_2 \in \mathcal{V}$ be arbitrary. Then:

$$|(T^\pi V_1)(s) - (T^\pi V_2)(s)|$$
$$= \Bigl|\sum_a \pi(a|s) \sum_{s'} P(s'|s,a) \gamma[V_1(s') - V_2(s')]\Bigr|$$
$$\leq \gamma \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) |V_1(s') - V_2(s')|$$
$$\leq \gamma \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) \|V_1 - V_2\|_\infty$$
$$= \gamma \cdot \|V_1 - V_2\|_\infty \cdot \underbrace{\sum_a \pi(a|s) \sum_{s'} P(s'|s,a)}_{=1}$$

Therefore:

$$\|T^\pi V_1 - T^\pi V_2\|_\infty \leq \gamma \|V_1 - V_2\|_\infty$$

**Conclusion:** $T^\pi$ is a $\gamma$-contraction. By Banach's theorem, iterating $V_{k+1} = T^\pi V_k$ from any $V_0$ converges to the **unique** fixed point $V^\pi$, satisfying $T^\pi V^\pi = V^\pi$ — i.e., the Bellman Expectation Equation. $\blacksquare$

---

## 4. Bellman Optimality Operator is a Contraction

Define the **Bellman Optimality Operator** $T^* : \mathcal{V} \to \mathcal{V}$:

$$(T^* V)(s) = \max_a \sum_{s'} P(s'|s,a)\bigl[R(s,a,s') + \gamma\,V(s')\bigr]$$

**Claim:** $T^*$ is also a $\gamma$-contraction.

**Proof:**

Use the identity $|\max_a f(a) - \max_a g(a)| \leq \max_a |f(a) - g(a)|$:

$$|(T^* V_1)(s) - (T^* V_2)(s)|$$
$$\leq \max_a \Bigl|\sum_{s'} P(s'|s,a)\,\gamma [V_1(s') - V_2(s')]\Bigr|$$
$$\leq \gamma \max_a \sum_{s'} P(s'|s,a) |V_1(s') - V_2(s')|$$
$$\leq \gamma \|V_1 - V_2\|_\infty$$

Therefore $\|T^* V_1 - T^* V_2\|_\infty \leq \gamma \|V_1 - V_2\|_\infty$.

**Conclusion:** $T^*$ has a unique fixed point $V^*$ satisfying $T^* V^* = V^*$ — the **Bellman Optimality Equation**. Value Iteration ($V_{k+1} = T^* V_k$) converges to $V^*$ with:

$$\|V_k - V^*\|_\infty \leq \gamma^k \|V_0 - V^*\|_\infty \quad \blacksquare$$

---

## 5. Greedy Policy Iteration Converges to Optimal

**Theorem.** Policy iteration (alternating policy evaluation + greedy improvement) converges to $\pi^*$ in finite steps.

**Sketch of proof:**

1. **Monotone improvement:** After greedy improvement, $V^{\pi'} \geq V^\pi$ pointwise.  
   *(Proof: $V^{\pi'}(s) \geq Q^\pi(s, \pi'(s)) = \max_a Q^\pi(s,a) \geq V^\pi(s)$.)*

2. **Finiteness:** There are finitely many deterministic policies ($|\mathcal{A}|^{|S|}$ in total).

3. **Monotone + Finite → Converges:** The sequence $V^{\pi_0} \leq V^{\pi_1} \leq \cdots$ must terminate. At termination $\pi^{k+1} = \pi^k$, meaning:

$$V^{\pi^k}(s) = \max_a Q^{\pi^k}(s,a) \quad \forall s$$

This is exactly the Bellman Optimality Equation, so $V^{\pi^k} = V^*$ and $\pi^k = \pi^*$. $\blacksquare$

---

## 6. Summary of Convergence Guarantees

| Algorithm | Operator | Fixed Point | Rate |
|-----------|----------|------------|------|
| Policy Evaluation | $T^\pi$ | $V^\pi$ | $O(\gamma^k)$ |
| Value Iteration | $T^*$ | $V^*$ | $O(\gamma^k)$ |
| Policy Iteration | Greedy step | $\pi^*$ | Finite steps |
