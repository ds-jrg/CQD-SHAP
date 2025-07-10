# Explainable Graph Query Answering

## Caching

The methodology that CQD uses to compute the answers to a multihop query is based on the results of each atom in the query. This means that we only need a link prediction model to answer each atom (`1p` queries) and then combine the results of each atom via the t-norm and t-conorm operators to compute the final answer set. As the number of entities and relations are finite, we can compute the answers for each possible `(entity, relation, ?)` triple and store them in a cache. Then, we can answer any complex query by just looking up the answers for each atom in the cache. This will significantly speed up the query answering process, and therefore the explanation process.

There is a python script `cqd_cache.py` that implements the caching mechanism. Note that this code is written just for the FB15k-237 dataset, and if you want to use it for other datasets, you need to modify the data directory in the code. Furthermore, there are various parameters that you can set in the code. The most important one is the `k` which is the number of top answers that you want to store in the cache. Another important parameter is `chunk_size`, which is the number of queries that will be processed in each call to the CQD model. As inputing all queries at once to the CQD model may lead to memory issues, you can set this parameter to a smaller value (e.g., 10000) based on your machine's memory capacity.

The final merged cache will be stored in `data/FB15k-237/all_1p_queries.json`. You can run the script as follows:

```bash
python cqd_cache.py
```