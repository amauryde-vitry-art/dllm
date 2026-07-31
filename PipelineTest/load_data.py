from datasets import load_dataset



def load_triviaqa(num_samples=100, seed=42):
    ds = load_dataset("trivia_qa", "rc", split="train")
    subset = ds.shuffle(seed=seed).select(range(num_samples))
    data = []
    labels = []
    for item in subset:
        question = item['question']
        data.append([{"role": "user", 'content': question}])
        labels.append({"question": question, "label": item['answer']['aliases']})
    return data, labels


def load_naturalquestion(num_samples=100, seed=42):
    ds = load_dataset("google-research-datasets/nq_open", split="train")
    subset = ds.shuffle(seed=seed).select(range(num_samples))
    data = []
    labels = []
    for item in subset:
        question = item['question']
        # nq_open provides a list of answer strings directly
        aliases = item['answer']
        data.append([{"role": "user", 'content': question}])
        labels.append({"question": question, "label": aliases})
    return data, labels


def load_hotpotqa(num_samples=100, seed=42):
    ds = load_dataset("hotpot_qa", "fullwiki", split="train")
    subset = ds.shuffle(seed=seed).select(range(num_samples))
    data = []
    labels = []
    for item in subset:
        question = item['question']
        answer = item['answer']
        data.append([{"role": "user", 'content': question}])
        labels.append({"question": question, "label": [answer]})
    return data, labels


