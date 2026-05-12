import collections
# 示例语料库保持一致
# 马可夫假设 计算整个句子的近似概率
corpus = "datawhale agent learns datawhale agent works"
tokens = corpus.split()
total_tokens = len(tokens)
# ----第一步：计算 P(datawhalew)---
count_datawhale = tokens.count("datawhale")
p_datawhale = count_datawhale / total_tokens
print(f"第一步：P(datawhale) ={count_datawhale}/{total_tokens} = {p_datawhale:.3f}")
# ----第二步：计算 P(agent/datawhale)----
# 先计算 bigrams 用于后续步骤
bigrams = zip(tokens, tokens[1:])
bigram_counts = collections.Counter(bigrams)
count_datawhale_agent = bigram_counts[('datawhale', 'agent')]
# count_datawhale 已在第一步计算
p_agent_given_datawhale = count_datawhale_agent / count_datawhale
print(f"第二步:P(agent|datawhale) ={count_datawhale_agent}/{count_datawhale} ={p_agent_given_datawhale:.3f})")
# ----第三步：计算 P(learns|agent)
count_agent_learns = bigram_counts[('agent', 'learns')]
count_agent = tokens.count('agent')
p_learns_given_agent = count_agent_learns / count_agent
print(f"第三步： P(learns|agent) ={count_agent_learns}/{count_agent} ={p_learns_given_agent:.3f}")
# ----最后：将概率连乘----
p_sentence = p_datawhale * p_agent_given_datawhale * p_learns_given_agent
print(
    f"最后: P('datawhale agent learns') = {p_datawhale: .3f} * {p_agent_given_datawhale: .3f} * {p_learns_given_agent: .3f} = {p_sentence: .3f}")
# N-gram 模型虽然简单有效，但有两个致命缺陷：
#
# 数据稀疏性 (Sparsity) ：如果一个词序列从未在语料库中出现，其概率估计就为 0，这显然是不合理的。虽然可以通过平滑 (Smoothing) 技术缓解，但无法根除。
# 泛化能力差：模型无法理解词与词之间的语义相似性。例如，即使模型在语料库中见过很多次 agent learns，它也无法将这个知识泛化到语义相似的词上。当我们计算 robot learns 的概率时，如果 robot 这个词从未出现过，或者 robot learns 这个组合从未出现过，模型计算出的概率也会是零。模型无法理解 agent 和 robot 在语义上的相似性