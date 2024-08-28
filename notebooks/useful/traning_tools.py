# Bibliotecas padrão
from datetime import datetime as dt
from collections import Counter

# Bibliotecas de terceiros
## Utilitárias
import numpy as np
from tqdm.notebook import tqdm
## Bibliotecas de aprendizado de máquina e análise de dados
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import BaggingClassifier

# Importações de constantes e configurações personalizadas
from useful.constants import N_ITER, SEED, TRAINVAL_SPLITS


def validate_model(model, X, y, param_grid, targets_map={}):
    skf = StratifiedKFold(n_splits=N_ITER, random_state=SEED, shuffle=True)
    skf_folds = skf.split(X, y)
    
    runs_metrics = {}
    for n, (trainval_idx, test_idx) in enumerate(tqdm(skf_folds, total=N_ITER)):
        X_trainval = X[trainval_idx]
        y_trainval = y[trainval_idx]
        
        X_test = X[test_idx]
        y_test = y[test_idx]
    
        best_params = get_best_params(model, X_trainval, y_trainval, param_grid)

        t0_t = dt.now()
        model.set_params(**best_params)
        model.fit(X_trainval, y_trainval)    
        tf_t = dt.now()
        tt_delta = (tf_t - t0_t).total_seconds()
        
        model_metrics = evaluate_model_performance(model, X_test, y_test, targets_map=targets_map)
        model_metrics['best_params'] = best_params
        model_metrics['training_time'] = tt_delta
        runs_metrics[n] = model_metrics
    
    runs_metrics = aggregate_run_metrics(runs_metrics, targets_map)
    return runs_metrics


def get_best_params(model, X_trainval, y_trainval, param_grid):
    skf = StratifiedKFold(n_splits=TRAINVAL_SPLITS, random_state=SEED, shuffle=True)
    
    grid_search = GridSearchCV(model, param_grid=param_grid, refit=False, cv=skf, n_jobs=-1)
    grid_search.fit(X_trainval, y_trainval)
    
    best_params = grid_search.best_params_

    return best_params


def evaluate_model_performance(model, X, y, targets_map={}):
    t0_p = dt.now()
    y_pred = model.predict(X)
    tf_p = dt.now()
    tp_delta = (tf_p - t0_p).total_seconds()
    
    cm = confusion_matrix(y, y_pred)

    if targets_map != {}:
        target_names = [targets_map[yy] for yy in np.unique(y)]

    else:
        target_names = None
            
    report_dict = classification_report(y, y_pred, output_dict=True, target_names=target_names)
    report_dict['cm'] = cm
    report_dict['prediction_time'] = tp_delta

    return report_dict


def aggregate_run_metrics(runs_res, targets_map):
    runs_ids = runs_res.keys()
    labels = targets_map.values()

    accuracies = [runs_res[i]['accuracy'] for i in runs_ids]
    cms = [runs_res[i]['cm'] for i in runs_ids]
    f1_scores = {i: [runs_res[j][i]['f1-score'] for j in runs_ids] for i in labels}
    recalls = {i: [runs_res[j][i]['recall'] for j in runs_ids] for i in labels}
    precisions = {i: [runs_res[j][i]['precision'] for j in runs_ids] for i in labels}
    best_params = [runs_res[i]['best_params'] for i in runs_ids]
    training_time = [runs_res[i]['training_time'] for i in runs_ids]
    prediction_time = [runs_res[i]['prediction_time'] for i in runs_ids]

    metrics = {
        'accuracies': accuracies,
        'cms': cms,
        'f1-scores': f1_scores,
        'recalls': recalls,
        'precisions': precisions,
        'best_params': best_params,
        'training_time': training_time,
        'prediction_time': prediction_time
      }

    return metrics



def print_res(res):
    mean_acc = np.mean(res['accuracies']) * 100
    std_acc = np.std(res['accuracies'], ddof=1) * 100
    
    print(f"===> ACURÁCIA MÉDIA <===\n({mean_acc:.2f} ± {std_acc:.2f})%", end='\n\n')
    print(f"===> MATRIZ DE CONFUSÃO GERAL <===\n{np.sum(res['cms'], axis=0)}", end='\n\n')
    
    print('===> RECALL, PRECISION E F1-SCORE MÉDIO <===') 
    print(f"{'Target'.ljust(12)} | {'Recall'.ljust(12)} (%) | {'Precision'.ljust(12)} (%) | {'F1-Score'.ljust(12)} (%)")
    print('-'*70)
    mean_recalls = []
    mean_precisions = []
    mean_f1_scores = []
    for l in res['recalls'].keys():  # Igual para todos
        mean_recall = np.mean(res['recalls'][l]) * 100
        mean_precision = np.mean(res['precisions'][l]) * 100
        mean_f1_score = np.mean(res['f1-scores'][l]) * 100

        std_recall = np.std(res['recalls'][l], ddof=1) * 100
        std_precision = np.std(res['precisions'][l], ddof=1) * 100
        std_f1_score = np.std(res['f1-scores'][l], ddof=1) * 100

        mean_recalls.append(mean_recall)
        mean_precisions.append(mean_precision)
        mean_f1_scores.append(mean_f1_score)
        
        print(f'{l.ljust(12)} | '
              f'{(str(np.round(mean_recall, 2)) + " ± " + str(np.round(std_recall, 2))).ljust(16)} | '
              f'{(str(np.round(mean_precision, 2)) + " ± " + str(np.round(std_precision, 2))).ljust(16)} | '
              f'{(str(np.round(mean_f1_score, 2)) + " ± " + str(np.round(std_f1_score, 2))).ljust(16)}')

    std_recalls = np.std(mean_recalls, ddof=1)
    std_precisions = np.std(mean_precisions, ddof=1)
    std_f1_scores = np.std(mean_f1_scores, ddof=1)
    
    mean_recalls = np.mean(mean_recalls)
    mean_precisions = np.mean(mean_precisions)
    mean_f1_scores = np.mean(mean_f1_scores)
    
    print(f"\n===> MÉDIA DO RECALL MÉDIO <===\n({mean_recalls:.2f} ± {std_recalls:.2f})%", end='\n\n')
    print(f"===> MÉDIA DO PRECISION MÉDIO <===\n({mean_precisions:.2f} ± {std_precisions:.2f})%", end='\n\n')
    print(f"===> MÉDIA DO F1-SCORE MÉDIO <===\n({mean_f1_scores:.2f} ± {std_f1_scores:.2f})%", end='\n\n')

    print('\n===> MELHORES HIPERPARÂMETROS <===') 
    print(f"{'Ocorrências'.ljust(12)} | {'Valores'.ljust(75)}")
    print('-'*130)
    params_counts = Counter(tuple(param.items()) for param in res['best_params'])
    params_counts_mc = params_counts.most_common()
    
    for pcm in params_counts_mc:  # [:5] para mostrar apenas top 5
        pcm_values = pcm[0]
        pcm_occ = pcm[1]
    
        print(f'{str(pcm_occ).ljust(12)} | {pcm_values}')

    mean_training_time = np.mean(res['training_time'])
    std_training_time = np.std(res['training_time'], ddof=1)
    
    mean_prediction_time = np.mean(res['prediction_time'])
    std_prediction_time = np.std(res['prediction_time'], ddof=1)
    
    print(f"\n===> TEMPO DE TREINAMENTO MÉDIO <===\n({mean_training_time:.4E} ± {std_training_time:.4E})s")
    print(f"\n===> TEMPO DE INFERÊNCIA MÉDIO <===\n({mean_prediction_time:.4E} ± {std_prediction_time:.4E})s")


def compute_feasibility(base_model, X, y, n_runs=3, n_estimators=100):
    skf = StratifiedKFold(n_splits=N_ITER, random_state=SEED, shuffle=True)
    skf_folds = skf.split(X, y)
    train_idx, test_idx = next(skf_folds)  # Apenas pava avaliar uma das runs do problema original

    bagging_model = BaggingClassifier(base_model, n_estimators=n_estimators, random_state=SEED)

    fit_times = []
    pred_times = []
    for _ in range(n_runs):
        X_train = X[train_idx]
        y_train = y[train_idx]
        
        X_test = X[test_idx]
        # y_test = y[test_idx]
                
        t0_f = dt.now()
        bagging_model.fit(X_train, y_train)
        tf_f = dt.now()
        tf_delta = (tf_f - t0_f).total_seconds()
        fit_times.append(tf_delta)

        t0_p = dt.now()
        _ = bagging_model.predict(X_test)  # Não associado para nenhuma variável pois só queremos avaliar o tempo, não o resultado
        tf_p = dt.now()
        tp_delta = (tf_p - t0_p).total_seconds()
        pred_times.append(tp_delta)

    feasibility_results = {
        'mean_pred_time': np.mean(pred_times),
        'std_pred_time': np.std(pred_times, ddof=1),
        'mean_fit_time': np.mean(fit_times),
        'std_fit_time': np.std(fit_times, ddof=1)
    }

    return feasibility_results


def get_mean_f1s(res):    
    f1s = [np.mean([res['f1-scores'][k][n] for k in res['f1-scores'].keys()]) for n in range(N_ITER)]

    return f1s