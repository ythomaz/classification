# Importação da base de dados
from sklearn.datasets import load_wine

# Bibliotecas para manipulação e visualização
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Bibliotecas para classificação
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# Bibliotecas para analise dos modelos
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.inspection import DecisionBoundaryDisplay


# Importando os dados organizando
dataset = load_wine()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df["CLASS"] = dataset.target

# Análise iniciais do dataset:
# print(dataset.DESCR)
# corr = df.corr()
# print(corr)

# Criando um dataframe apenas com um subconjunto de atributos escolhidos para análise com base na correlação
X = pd.DataFrame(np.c_[df['flavanoids'], df['proline']], columns=['flavanoids', 'proline'])
y = df["CLASS"]
data = pd.concat([X, y], axis=1)

# Visualizando o conjunto de dados
plt.figure(figsize=(8, 4))
sns.scatterplot(x='flavanoids', y='proline', data=data, hue='CLASS', palette="deep"
                ).set(title='Wine recognition dataset')
plt.show()

# Split da base em subconjuntos de treino e teste
x_treino, x_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3, random_state=0)

# Treinando os classificadores
KNN = KNeighborsClassifier(n_neighbors=5)
LR = LogisticRegression(max_iter=500, random_state=0)
SVML = SVC(kernel="linear", C=0.025)
SVM = SVC(gamma=2, C=1)
GAU = GaussianProcessClassifier(1.0 * RBF(1.0))
TREE = DecisionTreeClassifier(max_depth=5)
FOREST = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
ADA = AdaBoostClassifier()
NAIVE = GaussianNB()
QDA = QuadraticDiscriminantAnalysis()

KNN.fit(x_treino, y_treino)
LR.fit(x_treino, y_treino)
SVML.fit(x_treino, y_treino)
SVM.fit(x_treino, y_treino)
GAU.fit(x_treino, y_treino)
TREE.fit(x_treino, y_treino)
FOREST.fit(x_treino, y_treino)
ADA.fit(x_treino, y_treino)
NAIVE.fit(x_treino, y_treino)
QDA.fit(x_treino, y_treino)

# Definindo lista de modelos
clfs = [KNN, LR, SVML, SVM, GAU, TREE, FOREST, ADA, NAIVE, QDA]


# Função que chama os modelos treinados e plota os gráficos para análise

def comparacao(clfs):
    for clf in clfs:
        # configurações de exibição dos gráficos
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 3))
        fig.suptitle('Modelo %s' % str(clf), fontsize=16, y=1.08)
        ax1.set_title('Relatório de classificação')
        ax2.set_title('Matriz de confusão')
        ax3.set_title('Visualização do limite de decisão')

        # realização das predições na base de teste
        predicoes = clf.predict(x_teste)

        # criação do relatório de classificação e exibição em formato de mapa de calor
        clf_report = classification_report(y_teste, predicoes, labels=LR.classes_, output_dict=True)
        sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True, cmap="Spectral", ax=ax1)

        # criação e exibição da matriz de confusão
        cm = confusion_matrix(y_teste, predicoes, labels=LR.classes_)
        disp_matrix = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp_matrix.plot(ax=ax2)

        # criação e exibição dos limites de decisão do modelos
        disp = DecisionBoundaryDisplay.from_estimator(clf, x_teste, response_method="predict", xlabel='flavanoids',
                                                      ylabel='proline', alpha=0.5, ax=ax3)
        disp.ax_.scatter(X.loc[:, :'flavanoids'], X.loc[:, 'proline':], c=y, edgecolor="k")

    plt.show()

# Plotagem dos gráficos


comparacao(clfs)
