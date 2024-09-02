## 5. **Zastosowanie Wyboru Cech w Różnych Typach Modeli**

Wybór cech to kluczowy element każdego procesu modelowania w uczeniu maszynowym. Jakość wybranych cech bezpośrednio wpływa na wydajność, dokładność i interpretowalność modelu. W zależności od typu modelu, różne metody selekcji cech mogą być bardziej lub mniej efektywne. W tej sekcji omówimy zastosowanie wyboru cech w kontekście czterech głównych typów modeli: modeli liniowych, modeli drzewiastych, modeli opartych na kernelach oraz modeli sieci neuronowych.

#### Modele liniowe (np. regresja liniowa)

##### Wprowadzenie

Modele liniowe, takie jak regresja liniowa czy regresja logistyczna, są jednymi z najprostszych, ale zarazem najpotężniejszych narzędzi w uczeniu maszynowym. W modelach liniowych, wynik (zmienna zależna) jest liniową kombinacją cech (zmiennych niezależnych). Oznacza to, że każda cecha ma przypisaną wagę, która wskazuje, w jakim stopniu dana cecha wpływa na przewidywany wynik.

Formuła modelu liniowego może być zapisana jako:

$$
\hat{y} = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_n x_n
$$

Gdzie:
- $\hat{y}$ to przewidywana wartość,
- $w_0$ to wyraz wolny (bias),
- $w_1, w_2, \dots, w_n$ to wagi cech,
- $x_1, x_2, \dots, x_n$ to cechy (zmienne niezależne).

##### Znaczenie wyboru cech w modelach liniowych

W modelach liniowych, wybór cech ma ogromne znaczenie, ponieważ każda niepotrzebna cecha dodaje złożoność do modelu, co może prowadzić do nadmiernego dopasowania (overfitting). Co więcej, wagi przypisane cechom mogą być trudne do interpretacji, jeśli model zawiera zbyt wiele cech, zwłaszcza tych, które są ze sobą skorelowane (problem kolinearności).

##### Metody selekcji cech

1. **Eliminacja krokowa (Stepwise selection):**

Eliminacja krokowa to iteracyjna procedura, która dodaje lub usuwa cechy z modelu na podstawie ich statystycznej istotności. Proces ten może być przeprowadzony jako:

- **Forward selection**: Zaczyna się od modelu bez cech, a następnie dodaje się cechy, które najbardziej poprawiają model.
- **Backward elimination**: Zaczyna się od pełnego modelu, a następnie usuwa się cechy, które najmniej przyczyniają się do wyjaśnienia zmienności w danych.

Przykładowy kod:

```python
import statsmodels.api as sm

# Dane przykładowe
X = df_standardized
y = y

# Dodanie stałej (bias) do modelu
X_with_constant = sm.add_constant(X)

# Początkowy pełny model
model = sm.OLS(y, X_with_constant).fit()

# Backward elimination
def backward_elimination(data, target, significance_level=0.05):
    features = data.columns.tolist()
    while len(features) > 0:
        X_with_constant = sm.add_constant(data[features])
        model = sm.OLS(target, X_with_constant).fit()
        max_p_value = model.pvalues.max()
        if max_p_value > significance_level:
            excluded_feature = model.pvalues.idxmax()
            features.remove(excluded_feature)
        else:
            break
    return model, features

model, selected_features = backward_elimination(X, y)
print("Wybrane cechy:", selected_features)
```

2. **Regularizacja LASSO:**

LASSO (Least Absolute Shrinkage and Selection Operator) to technika regularizacji, która może zmniejszyć współczynniki niektórych cech do zera, efektywnie eliminując je z modelu.

Przykładowy kod:

```python
from sklearn.linear_model import Lasso

# LASSO
lasso = Lasso(alpha=0.1)
lasso.fit(X, y)

# Wagi cech
print("Wagi cech po LASSO:", lasso.coef_)
```

##### Przykład zastosowania

Załóżmy, że mamy zbiór danych dotyczący cen nieruchomości. W tym przypadku możemy zastosować LASSO, aby wyeliminować cechy, które nie mają istotnego wpływu na cenę nieruchomości, takie jak odległość od najbliższego sklepu spożywczego, jeśli inne cechy, takie jak lokalizacja i metraż, są znacznie ważniejsze.

#### Modele drzewiaste (np. drzewa decyzyjne, lasy losowe)

##### Wprowadzenie

Modele drzewiaste, takie jak drzewa decyzyjne, lasy losowe (Random Forests) i gradient boosting, są powszechnie stosowane w uczeniu maszynowym ze względu na ich zdolność do radzenia sobie zarówno z danymi strukturalnymi, jak i nieliniowymi. Modele te są bardziej elastyczne niż modele liniowe, ponieważ potrafią uchwycić złożone zależności między cechami a zmienną docelową.

##### Znaczenie wyboru cech w modelach drzewiastych

W modelach drzewiastych wybór cech jest automatycznie wbudowany w proces trenowania. Drzewa decyzyjne same wybierają te cechy, które najlepiej dzielą dane na podstawie kryteriów takich jak indeks Gini, entropia, czy redukcja wariancji. Jednakże, zmniejszenie liczby cech przed trenowaniem modelu drzewiastego może nadal przynieść korzyści, takie jak skrócenie czasu treningu i redukcja ryzyka nadmiernego dopasowania.

##### Metody selekcji cech

1. **Ważność cech (Feature importance):**

Ważność cech w modelach drzewiastych jest mierzona na podstawie ich wpływu na podziały drzewa. Cechy, które są częściej używane do tworzenia podziałów i które bardziej poprawiają jakość podziału (np. zmniejszają nieczystość), są uznawane za bardziej istotne.

Przykładowy kod:

```python
from sklearn.ensemble import RandomForestClassifier

# Tworzenie modelu lasów losowych
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X, y)

# Ważność cech
importances = rf.feature_importances_

# Sortowanie ważności cech
indices = np.argsort(importances)[::-1]

# Wyświetlanie wyników
for i in range(X.shape[1]):
    print(f"Cechy {indices[i]}: Ważność = {importances[indices[i]]}")

# Wizualizacja wyników
plt.figure(figsize=(10, 6))
plt.title("Znaczenie cech w lasach losowych")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.show()
```

2. **Przycinanie drzewa (Tree pruning):**

Przycinanie drzewa to technika, która redukuje rozmiar drzewa decyzyjnego poprzez usuwanie gałęzi, które mają niewielki wpływ na model. W ten sposób eliminujemy cechy, które prowadzą do nadmiernego dopasowania.

Przykładowy kod:

```python
from sklearn.tree import DecisionTreeClassifier

# Drzewo decyzyjne z ograniczeniem maksymalnej głębokości (przycinanie)
tree = DecisionTreeClassifier(max_depth=3)
tree.fit(X, y)

# Wizualizacja drzewa
from sklearn import tree
plt.figure(figsize=(12, 8))
tree.plot_tree(tree, feature_names=X.columns, filled=True)
plt.show()
```

##### Przykład zastosowania

Załóżmy, że pracujemy z danymi dotyczącymi kredytów bankowych. W tym przypadku model lasów losowych może automatycznie przypisać większą wagę cechom, które najlepiej przewidują ryzyko niespłacenia kredytu, takim jak historia kredytowa czy dochody, podczas gdy cechy mniej istotne, jak liczba dzieci, mogą zostać uznane za mniej ważne.

#### Modele oparte na kernelach (np. SVM)

##### Wprowadzenie

Modele oparte na kernelach, takie jak maszyny wektorów nośnych (SVM), są potężnymi narzędziami do klasyfikacji i regresji. Zastosowanie kerneli pozwala na przekształcenie danych do wyższego wymiaru, gdzie stają się one liniowo separowalne. Dzięki temu SVM potrafią uchwycić złożone,

 nieliniowe zależności w danych.

##### Znaczenie wyboru cech w modelach opartych na kernelach

Chociaż SVM są potężnymi narzędziami, ich skuteczność może być znacznie ograniczona przez nadmiar cech, szczególnie gdy wiele z tych cech nie wnosi istotnej informacji. Zbyt wiele cech może prowadzić do nadmiernego dopasowania, a także znacznie zwiększyć złożoność obliczeniową.

##### Metody selekcji cech

1. **Recursive Feature Elimination (RFE):**

RFE jest iteracyjną metodą selekcji cech, która trenowuje model na pełnym zestawie cech, a następnie stopniowo usuwa najmniej istotne cechy. Proces ten jest powtarzany, aż pozostanie optymalny podzbiór cech.

Przykładowy kod:

```python
from sklearn.feature_selection import RFE
from sklearn.svm import SVC

# SVM z jądrem liniowym
svm = SVC(kernel="linear")

# RFE
selector = RFE(svm, n_features_to_select=5, step=1)
selector = selector.fit(X, y)

# Wybrane cechy
print("Ranking cech (RFE):", selector.ranking_)
print("Wybrane cechy:", X.columns[selector.support_])
```

2. **Ważność cech w SVM z jądrem liniowym:**

W przypadku SVM z jądrem liniowym, wagi przypisane cechom mogą być interpretowane podobnie jak w regresji liniowej, co umożliwia bezpośrednie zidentyfikowanie istotnych cech.

Przykładowy kod:

```python
# SVM z jądrem liniowym
svm_linear = SVC(kernel="linear")
svm_linear.fit(X, y)

# Wagi cech
coef = svm_linear.coef_[0]

# Sortowanie wag
indices = np.argsort(np.abs(coef))[::-1]

# Wyświetlanie wyników
for i in range(X.shape[1]):
    print(f"Cechy {indices[i]}: Waga = {coef[indices[i]]}")

# Wizualizacja wyników
plt.figure(figsize=(10, 6))
plt.title("Wagi cech w SVM z jądrem liniowym")
plt.bar(range(X.shape[1]), coef[indices], align="center")
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.show()
```

##### Przykład zastosowania

Załóżmy, że mamy dane dotyczące klasyfikacji chorób na podstawie wyników badań medycznych. W tym przypadku, zastosowanie RFE z SVM pozwoli na zidentyfikowanie tych cech (np. poziom cholesterolu, ciśnienie krwi), które są najbardziej istotne dla diagnozy.

#### Modele sieci neuronowych

##### Wprowadzenie

Sieci neuronowe to klasa modeli, która jest inspirowana działaniem ludzkiego mózgu. Dzięki zdolności do modelowania złożonych nieliniowych relacji, sieci neuronowe są szczególnie skuteczne w rozpoznawaniu obrazów, przetwarzaniu języka naturalnego, analizie sygnałów i wielu innych dziedzinach.

##### Znaczenie wyboru cech w sieciach neuronowych

W sieciach neuronowych wybór cech może być nieco mniej krytyczny niż w przypadku modeli liniowych czy SVM, ponieważ sieci neuronowe potrafią samodzielnie uczyć się odpowiednich reprezentacji cech na poziomie ukrytych warstw. Niemniej jednak, nadmiar nieistotnych cech może zwiększyć ryzyko nadmiernego dopasowania i znacząco wydłużyć czas treningu. Dlatego też wciąż istnieje potrzeba selekcji cech przed trenowaniem sieci neuronowej.

##### Metody selekcji cech

1. **Regularizacja L1 i L2:**

Regularizacja L1 (LASSO) i L2 (Ridge) może być stosowana w sieciach neuronowych w celu penalizacji dużych wag, co pomaga w redukcji nadmiernego dopasowania oraz może skutkować wyeliminowaniem nieistotnych cech.

Przykładowy kod:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l1, l2

# Tworzenie sieci neuronowej z regularizacją L1 i L2
model = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],), kernel_regularizer=l1(0.001)),
    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    Dense(1, activation='sigmoid')
])

# Kompilowanie modelu
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Trenowanie modelu
model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)
```

2. **Dropout:**

Dropout to technika regularizacyjna, która losowo "wyłącza" (czyli ignoruje) pewien procent neuronów w trakcie treningu, co zapobiega nadmiernemu dopasowaniu i zmusza sieć do nauki bardziej robustnych cech.

Przykładowy kod:

```python
from tensorflow.keras.layers import Dropout

# Tworzenie sieci neuronowej z Dropout
model = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Kompilowanie modelu
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Trenowanie modelu
model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)
```

3. **Feature Importance with Permutation Importance:**

Permutation importance to technika, która mierzy, jak degradacja modelu następuje, gdy wartości danej cechy są losowo permutowane, co pozwala na ocenę jej znaczenia.

Przykładowy kod:

```python
import tensorflow as tf
from sklearn.inspection import permutation_importance

# Tworzenie i trenowanie modelu sieci neuronowej
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

# Użycie permutacji do oceny ważności cech
def model_predict(X):
    return model.predict(X).flatten()

perm_importance = permutation_importance(model_predict, X, y, n_repeats=10)
importance_scores = perm_importance.importances_mean

# Wyświetlanie wyników
print("Znaczenie cech (permutacja):", importance_scores)
```

##### Przykład zastosowania

Załóżmy, że mamy dane z obrazami, na których musimy rozpoznać obiekty. W takim przypadku sieć neuronowa może automatycznie nauczyć się odpowiednich cech (np. krawędzi, tekstur) z surowych danych obrazowych. Regularizacja L1/L2 oraz dropout pomogą uniknąć nadmiernego dopasowania, zwłaszcza gdy mamy ograniczoną ilość danych treningowych.

---

© 2024 Marian Witkowski - wszelkie prawa zastrzeżone
