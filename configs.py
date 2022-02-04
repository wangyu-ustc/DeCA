DeCA_configs = {
 ('GMF', "ml-100k"): {'iterative': True,
   'C_2': 1000,
   'C_1': 1000,
   'epochs': 20,
   'early_stop': False,
   'lambda0': 0.1,
   'alpha': 1},
 ('NeuMF-end', "ml-100k"): {'iterative': True,
   'C_2': 10,
   'C_1': 1000,
   'epochs': 100,
   'early_stop': False,
   'lambda0': 0.01,
   'alpha': 1},
 ('CDAE', "ml-100k"): {'iterative': True,
   'C_2': 100,
   'C_1': 1000,
   'epochs': 400,
   'early_stop': False,
   'lambda0': 1.0,
   'alpha': 0.5},
 ('LightGCN', "ml-100k"): {'iterative': True,
   'C_2': 100,
   'C_1': 100,
   'epochs': 150,
   'early_stop': True,
   'lambda0': 0.1,
   'alpha': 0.0},
 ('GMF', 'modcloth'): {'iterative': True,
  'C_2': 10,
  'C_1': 100,
  'epochs': 40,
  'early_stop': False,
  'lambda0': 0.1,
  'alpha': 0},
 ('NeuMF-end', 'modcloth'): {'iterative': True,
  'C_2': 10,
  'C_1': 1000,
  'epochs': 40,
  'early_stop': False,
  'lambda0': 10,
  'alpha': 0.5},
 ('CDAE', 'modcloth'): {'iterative': True,
  'C_2': 100,
  'C_1': 1000,
  'epochs': 40,
  'early_stop': True,
  'lambda0': 1,
  'alpha': 0},
 ('LightGCN', 'modcloth'): {'iterative': True,
  'C_2': 10,
  'C_1': 1000,
  'epochs': 60,
  'early_stop': False,
  'lambda0': 0.1,
  'alpha': 0},
 ('GMF', 'adressa'): {'iterative': True,
  'C_2': 10,
  'C_1': 100,
  'epochs': 15,
  'early_stop': False,
  'lambda0': 0.0,
  'alpha': 1},
 ('NeuMF-end', 'adressa'): {'iterative': True,
  'C_2': 10,
  'C_1': 1000,
  'epochs': 30,
  'early_stop': True,
  'lambda0': 1,
  'alpha': 0},
 ('CDAE', 'adressa'): {'iterative': True,
  'C_2': 10,
  'C_1': 1000,
  'epochs': 50,
  'early_stop': False,
  'lambda0': 0,
  'alpha': 0},
 ('LightGCN', 'adressa'): {'iterative': True,
  'C_2': 1000,
  'C_1': 10,
  'epochs': 15,
  'early_stop': True,
  'lambda0': 10,
  'alpha': 0},
 ('GMF', 'electronics'): {'iterative': True,
  'C_2': 10,
  'C_1': 1000,
  'epochs': 15,
  'early_stop': False,
  'lambda0': 0.01,
  'alpha': 1},
 ('NeuMF-end', 'electronics'): {'iterative': True,
  'C_2': 10,
  'C_1': 100,
  'epochs': 15,
  'early_stop': False,
  'lambda0': 1000,
  'alpha': 0.5},
 ('CDAE', 'electronics'): {'iterative': True,
  'C_2': 10,
  'C_1': 100,
  'epochs': 15,
  'early_stop': True,
  'lambda0': 0.1,
  'alpha': 0},
 ('LightGCN', 'electronics'): {'iterative': True,
  'C_2': 10,
  'C_1': 1000,
  'epochs': 10,
  'early_stop': False,
  'lambda0': 0.01,
  'alpha': 1}}

DeCAp_configs = {
 ('GMF', "ml-100k"): {'iterative': True,
   'C_2': 1000,
   'C_1': 1000,
   'epochs': 20,
   'early_stop': False,
   'lambda0': 0.1,
   'alpha': 1},
 ('NeuMF-end', "ml-100k"): {'iterative': True,
   'C_2': 10,
   'C_1': 100,
   'epochs': 100,
   'early_stop': True,
   'lambda0': 0.0,
   'alpha': 1},
 ('CDAE', 'ml-100k'): {'iterative': True,
   'C_2': 10,
   'C_1': 1000,
   'epochs': 500,
   'early_stop': False,
   'lambda0': 1,
   'alpha': 1},
 ('LightGCN', "ml-100k"): {'iterative': True,
   'C_2': 1,
   'C_1': 1,
   'epochs': 150,
   'early_stop': True,
   'lambda0': 0.0,
   'alpha': 1.0},
 ('GMF', 'modcloth'): {'iterative': True,
  'C_2': 10,
  'C_1': 100,
  'epochs': 40,
  'early_stop': False,
  'lambda0': 0.1,
  'alpha': 1},
 ('NeuMF-end', 'modcloth'): {'iterative': True,
  'C_2': 10,
  'C_1': 1000,
  'epochs': 40,
  'early_stop': True,
  'lambda0': 10,
  'alpha': 1},
 ('CDAE', 'modcloth'): {'iterative': True,
  'C_2': 100,
  'C_1': 1000,
  'epochs': 40,
  'early_stop': True,
  'lambda0': 1,
  'alpha': 1},
 ('LightGCN', 'modcloth'): {'iterative': True,
  'C_2': 10,
  'C_1': 10,
  'epochs': 60,
  'early_stop': True,
  'lambda0': 0.1,
  'alpha': 0.5},
 ('GMF', 'adressa'): {'iterative': True,
  'C_2': 10,
  'C_1': 1000,
  'epochs': 10,
  'early_stop': False,
  'lambda0': 0.01,
  'alpha': 1},
 ('NeuMF-end', 'adressa'): {'iterative': True,
  'C_2': 10,
  'C_1': 100,
  'epochs': 30,
  'early_stop': True,
  'lambda0': 1,
  'alpha': 1},
 ('CDAE', 'adressa'): {'iterative': True,
  'C_2': 10,
  'C_1': 100,
  'epochs': 50,
  'early_stop': False,
  'lambda0': 0,
  'alpha': 1},
 ('LightGCN', 'adressa'): {'iterative': False,
  'C_2': 10,
  'C_1': 1000,
  'epochs': 15,
  'early_stop': True,
  'lambda0': 0.01,
  'alpha': 0.5},
 ('GMF', 'electronics'): {'iterative': True,
  'C_2': 10,
  'C_1': 1000,
  'epochs': 10,
  'early_stop': False,
  'lambda0': 0.01,
  'alpha': 1},
 ('NeuMF-end', 'electronics'): {'iterative': True,
  'C_2': 10,
  'C_1': 100,
  'epochs': 10,
  'early_stop': True,
  'lambda0': 1000,
  'alpha': 0},
 ('CDAE', 'electronics'): {'iterative': True,
  'C_2': 10,
  'C_1': 100,
  'epochs': 15,
  'early_stop': True,
  'lambda0': 0.1,
  'alpha': 0},
 ('LightGCN', 'electronics'): {'iterative': True,
  'C_2': 10,
  'C_1': 1000,
  'epochs': 10,
  'early_stop': False,
  'lambda0': 0.01,
  'alpha': 1}}


baseline_configs = {
 ('GMF', 'ml-100k'): {'epochs': 20, 'early_stop': True},
 ('NeuMF-end', 'ml-100k'): {'epochs': 100, 'early_stop': True},
 ('CDAE', 'ml-100k'): {'epochs': 200, 'early_stop': False},
 ("LightGCN", 'ml-100k'): {'epochs': 200, 'early_stop': True},
 ('GMF', 'modcloth'): {'epochs': 40, 'early_stop': True},
 ('NeuMF-end', 'modcloth'): {'epochs': 40, 'early_stop': True},
 ('CDAE', 'modcloth'): {'epochs': 40, 'early_stop': False},
 ('LightGCN', 'modcloth'): {'epochs': 40, 'early_stop': False},
 ('GMF', 'adressa'): {'epochs': 10, 'early_stop': True},
 ('NeuMF-end', 'adressa'): {'epochs': 10, 'early_stop': True},
 ('CDAE', 'adressa'): {'epochs': 50, 'early_stop': True},
 ('LightGCN', 'adressa'): {'epochs': 15, 'early_stop': False},
 ('GMF', 'electronics'): {'epochs': 10, 'early_stop': True},
 ('NeuMF-end', 'electronics'): {'epochs': 10, 'early_stop': False},
 ('CDAE', 'electronics'): {'epochs': 15, 'early_stop': False},
 ('LightGCN', 'electronics'): {'epochs': 10, 'early_stop': False}}


lambdas = {
 ('GMF', 'ml-100k', 'normal'): 0.1,
 ('NeuMF-end', 'ml-100k', 'normal'): 0.01,
 ('CDAE', 'ml-100k', 'normal'): 1.0,
 ('LightGCN', 'ml-100k', 'normal'): 0.01,
 ('GMF', 'modcloth', 'normal'): 0.1,
 ('NeuMF-end', 'modcloth', 'normal'): 10,
 ('CDAE', 'modcloth', 'normal'): 1,
 ('LightGCN', 'modcloth', 'normal'): 0.1,
 ("GMF", 'adressa', 'normal'): 0.01,
 ("NeuMF-end", 'adressa', 'normal'): 1,
 ('CDAE', 'adressa', 'normal'): 1,
 ('LightGCN', 'adressa', 'normal'): 0.01,
 ('NeuMF-end', 'electronics', 'normal'): 100,
 ('GMF', 'electronics', 'normal'): 0.01,
 ('CDAE', 'electronics', 'normal'): 0.1,
 ('LightGCN', 'electronics', 'normal'): 0.01,
}


def get_config(model, dataset, method):
 if method == 'DeCA':
  param = DeCA_configs[(model, dataset)]
 elif method == 'DeCAp':
  param = DeCAp_configs[(model, dataset)]
 else:
  param = baseline_configs[(model, dataset)]
  param['lambda0'] = lambdas[(model, dataset, 'normal')]
 return param