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


