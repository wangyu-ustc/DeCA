import numpy as np
import pandas as pd

type_2_number = {'FN': 1, 'FP': 2, 'FB': 3}
def process(name):
    if 'it' in name:
        train_method = 'iterative_in_both'
        name = name.replace("-it-", '-')
    else:
        train_method = 'normal'

    name = name.split("-")
    de_type = type_2_number[name[0]]
    prior = name[1]
    KL_direction = eval(name[2])
    if name[0] != 'FB':
        negative_C = positive_C = eval(name[3])
    else:
        positive_C = eval(name[3])
        negative_C = eval(name[4])
    for i, word in enumerate(name):
        if 'epoch' in word:
            epochs = int(word[-2:])
            early_stop = eval(name[i+1])
    return train_method, de_type, prior, KL_direction, positive_C, negative_C, epochs, early_stop

def get_top_k(dataset):
    return {
        "ml-100k": [3, 5, 10, 20],
        "modcloth": [3, 5, 10, 20],
        "adressa": [3, 5, 10, 20],
        "electronics": [5, 10, 20, 50]
    }[dataset]


def store(rows, titles, top_k, target_path):
    rows = np.concatenate(rows, axis=0)
    df = pd.DataFrame(
        {"method": titles,
         f"(clean)precision@{top_k[0]}": list(rows[:, 0]),
         f"(clean)precision@{top_k[1]}": list(rows[:, 1]),
         f"(clean)precision@{top_k[2]}": list(rows[:, 2]),
         f"(clean)precision@{top_k[3]}": list(rows[:, 3]),
         f"(clean)recall@{top_k[0]}": list(rows[:, 4]),
         f"(clean)recall@{top_k[1]}": list(rows[:, 5]),
         f"(clean)recall@{top_k[2]}": list(rows[:, 6]),
         f"(clean)recall@{top_k[3]}": list(rows[:, 7]),
         f"(clean)NDCG@{top_k[0]}": list(rows[:, 8]),
         f"(clean)NDCG@{top_k[1]}": list(rows[:, 9]),
         f"(clean)NDCG@{top_k[2]}": list(rows[:, 10]),
         f"(clean)NDCG@{top_k[3]}": list(rows[:, 11]),
         f"(clean)MRR@{top_k[0]}": list(rows[:, 12]),
         f"(clean)MRR@{top_k[1]}": list(rows[:, 13]),
         f"(clean)MRR@{top_k[2]}": list(rows[:, 14]),
         f"(clean)MRR@{top_k[3]}": list(rows[:, 15])
         }
    )
    df.to_csv(target_path, index=False, sep=',')
