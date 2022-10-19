import streamlit as st
import base64
import pandas as pd
import numpy as np
import xgboost as xgb
import re
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

from sklearn.metrics import roc_curve, roc_auc_score
from tqdm.notebook import tqdm_notebook
from stqdm import stqdm
from warnings import filterwarnings

st.set_option('deprecation.showPyplotGlobalUse', False)
st.title('Предиктивная модель')

st.markdown("""
Приложение для анализа больших данных, предсказание пола клиента на основе экономических транзакций
* **Библиотеки Python:** streamlit, pandas, xgboost, seaborn, matplotlib, sklearn
* **Источник данных:** [drive.google.com](https://drive.google.com/drive/folders/1qMa3qAz1stF6sqZ8zOPdzzqt-tM0gFfy?usp=sharing).
""")

st.sidebar.header('MCC-код')
cod = st.sidebar.text_input('Код категории', '5912')
search = st.sidebar.button('Найти')


def load_data(url, ixc):
    url = 'https://drive.google.com/uc?id=' + url.split('/')[-2]
    df = pd.read_csv(url, sep=';', index_col=ixc)
    return df


def load_t_data(url, ixc):
    url = 'https://drive.google.com/uc?id=' + url.split('/')[-2]
    df = pd.read_csv(url, index_col=ixc)
    return df

tran = load_t_data("https://drive.google.com/file/d/1hw02GjNhR-ibhGabUYAQ20__brjIj5CI/view?usp=sharing", 'customer_id')
mcc = load_data("https://drive.google.com/file/d/1yOffnB1BjuHnDBLrkuHZh4q6I6fKYJre/view?usp=sharing", 'mcc_code')
typ = load_data("https://drive.google.com/file/d/15WfmNBkepimKfFX7ojx7Ta2Jt8RhMs9d/view?usp=sharing", 'tr_type')
g_train = load_t_data("https://drive.google.com/file/d/1qmzzYmEi2RGbPQjY0cAjXUOWoBA52ndh/view?usp=sharing",
                      'customer_id')
g_test = load_t_data("https://drive.google.com/file/d/1scNWnpAHndNYgEPanQDO-Gsi4TYhRDqF/view?usp=sharing",
                     'customer_id')
tran_train = tran.join(g_train, how='inner')
tran_test = tran.join(g_test, how='inner')

fe = pd.merge(tran, g_train, on='customer_id', how='left')
fe = pd.merge(fe, typ, on='tr_type', how='inner')
fe = pd.merge(fe, mcc, on='mcc_code', how='inner')

if search:
    st.sidebar.write(mcc.loc[int(cod)]["mcc_description"])

del tran

fe1 = fe.loc[(fe['gender'] == 1) & (fe['amount'] < 0)]
fe0 = fe.loc[(fe['gender'] == 0) & (fe['amount'] < 0)]
a = fe1["amount"].mean() - fe0["amount"].mean()

params = {
    'eta': 0.1,
    'max_depth': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,

    'gamma': 1,
    'lambda': 3,
    'alpha': 0,
    'min_child_weight': 0,

    'eval_metric': 'auc',
    'objective': 'binary:logistic',
    'booster': 'gbtree',
    'njobs': -1,
    'tree_method': 'approx'
}


# Cross-validation score (среднее значение метрики ROC AUC на тренировочных данных)
def cv_score(params, train, y_true):
    cv_res = xgb.cv(params, xgb.DMatrix(train, y_true),
                    early_stopping_rounds=10, maximize=True,
                    num_boost_round=10000, nfold=5, stratified=True)
    index_argmax = cv_res['test-auc-mean'].argmax()
    st.write('Точность построенной модели: {:.3f}+-{:.3f}'.format(cv_res.loc[index_argmax]['test-auc-mean'],
                                                                  cv_res.loc[index_argmax]['test-auc-std']))


# Построение модели + возврат результатов классификации тестовых пользователей
def fit_predict(params, num_trees, train, test, target):
    params['learning_rate'] = params['eta']
    clf = xgb.train(params, xgb.DMatrix(train.values, target, feature_names=list(train.columns)),
                    num_boost_round=num_trees, maximize=True)
    y_pred = clf.predict(xgb.DMatrix(test.values, feature_names=list(train.columns)))
    submission = pd.DataFrame(index=test.index, data=y_pred, columns=['probability'])
    return clf, submission


# Отрисовка важности переменных. Важность переменной - количество разбиений выборки, 
# в которых участвует данная переменная
def draw_feature_importances(clf, top_k=10):
    plt.figure(figsize=(10, 10))

    importances = dict(sorted(clf.get_score().items(), key=lambda x: x[1])[-top_k:])
    y_pos = np.arange(len(importances))

    plt.barh(y_pos, list(importances.values()), align='center', color='green')
    plt.yticks(y_pos, importances.keys(), fontsize=12)
    plt.xticks(fontsize=12)
    plt.xlabel('Feature importance', fontsize=15)
    plt.title('Features importances, Gender Prediction', fontsize=18)
    plt.ylim(-0.5, len(importances) - 0.5)
    plt.show(block=False)


def plot_pivot_table(pivot_table):
    plt.figure(figsize=(10, 10))
    sns.heatmap(pivot_table, cmap="YlGnBu", annot=True,
                fmt='.3g', annot_kws={"size": 14})
    plt.xticks(fontsize=15)
    plt.yticks(rotation=0, fontsize=15)
    plt.xlabel('Bucket', size=18)
    plt.ylabel('Hour', fontsize=18)
    plt.title('Gender analysis per bucket and hour', fontsize=20)
    plt.show()


tqdm_notebook.pandas(desc="Progress:")


def features_creation_advanced(x):
    features = []

    if mcc_param:
        tr1 = tran_train['mcc_code'].unique()
        tr2 = tran_test['mcc_code'].unique()

        features.append(pd.Series(x[(x['mcc_code'].isin(tr1)) & (x['mcc_code'].isin(tr2))]['mcc_code'].value_counts(
            normalize=True).add_prefix('code_')))
        del tr1
        del tr2

    if time_param:
        x['day'] = x['tr_datetime'].str.split().apply(lambda t: int(t[0]) % 7)
        x['hour'] = x['tr_datetime'].apply(lambda t: re.search(' \d*', t).group(0)).astype(int)
        x['night'] = ~x['hour'].between(6, 22).astype(int)

        features.append(pd.Series(x['day'].value_counts(normalize=True).add_prefix('day_')))
        features.append(pd.Series(x['hour'].value_counts(normalize=True).add_prefix('hour_')))
        features.append(pd.Series(x['night'].value_counts(normalize=True).add_prefix('night_')))

    if math_param:
        features.append(pd.Series(x[x['amount'] > 0]['amount'].agg(['min', 'max', 'mean', 'median', 'std', 'count']) \
                                  .add_prefix('positive_transactions_')))
        features.append(pd.Series(x[x['amount'] < 0]['amount'].agg(['min', 'max', 'mean', 'median', 'std', 'count']) \
                                  .add_prefix('negative_transactions_')))
    return pd.concat(features)


with st.expander("Математика"):
    st.write('Разница между средними тратами женщин и мужчин: ' + str(abs(round(a, 2))) + ' у.е.')

with st.expander("Тепло"):
    st.header('Распределение трат по часам и сумме')
    fs = fe
    otric = fs[fs.amount < 0].amount
    fs['amount_bucket'] = pd.qcut(otric, 5, labels=['Very High', 'High', 'Middle', 'Low', 'Very Low'])
    fs['amount_bucket'] = fs['amount_bucket'].cat.add_categories('Income').fillna('Income')
    frameNew = pd.pivot_table(fs,
                              values='gender',
                              index=fs['tr_datetime'].apply(lambda x: x.split()[1].split(':')[0]),
                              columns=['amount_bucket'])
    plot_pivot_table(frameNew)
    st.pyplot()

with st.expander("Категории"):
    st.header('Наиболее характерные траты для того или иного пола')
    fr = fe
    fr[['tr_day', 'time']] = fr.tr_datetime.str.split(expand=True)
    f = lambda x: x.mcc_code.nunique() >= 75
    fr = fr.groupby('tr_day').filter(f)

    fr = fr.groupby(['mcc_description', 'gender']).amount.count()
    fr = fr.groupby(['mcc_description']).diff()
    fr = fr.reindex(fr.abs().sort_values(ascending=False).index)
    res = [fr.index.tolist()[i][0] for i in range(20)]
    for i in range(len(res)):
        st.write(res[i])

math_param = st.checkbox('Математические параметры', value=True)
time_param = st.checkbox('Временные параметры', value=True)
mcc_param = st.checkbox('Параметры категорий', value=True)

stqdm.pandas()

st.write('Обучить и рассчитать:')
go = st.button('Вперёд!')


def convert_df(df):
    return df.to_csv().encode('utf-8')


if go:
    # matplotlib.use('TkAgg')
    st.header('Модель и наиболее важные критерии')
    data_train = tran_train.groupby(tran_train.index) \
        .progress_apply(features_creation_advanced).unstack(-1)
    data_test = tran_test.groupby(tran_test.index) \
        .progress_apply(features_creation_advanced).unstack(-1)

    target = data_train.join(g_train, how='inner')['gender']
    cv_score(params, data_train, target)

    clf, submission = fit_predict(params, 70, data_train, data_test, target)
    draw_feature_importances(clf, 10)
    st.pyplot()

    csv = convert_df(submission)

    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='submission_advanced.csv',
        mime='text/csv',
    )


