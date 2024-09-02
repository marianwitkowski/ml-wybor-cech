## 3. **Kryteria Oceny Cech**

Ocena cech w procesie budowy modelu uczenia maszynowego jest kluczowa dla zrozumienia, które zmienne mają największy wpływ na wynik predykcji. W tym rozdziale omówimy trzy główne kryteria oceny cech: informację wzajemną, wagę cech w modelach liniowych oraz znaczenie cech w modelach nieliniowych. Zrozumienie tych kryteriów pozwala na bardziej świadome podejście do selekcji cech, co z kolei prowadzi do tworzenia bardziej efektywnych i interpretowalnych modeli.

#### Informacja wzajemna

Informacja wzajemna (ang. *mutual information*) jest miarą ilości informacji, jaką jedna zmienna niesie o drugiej. W kontekście uczenia maszynowego informacja wzajemna jest wykorzystywana do oceny zależności między cechą a zmienną docelową. Jest to miara nieliniowa, co oznacza, że może uchwycić zarówno związki liniowe, jak i nieliniowe między zmiennymi.

##### Definicja i wzór

Formalnie, informacja wzajemna $I(X; Y)$ między dwiema zmiennymi $X$ i $Y$ jest definiowana jako:

$$
I(X; Y) = \sum_{x \in X} \sum_{y \in Y} p(x, y) \log \left( \frac{p(x, y)}{p(x) p(y)} \right)
$$

Gdzie:
- $p(x, y)$ jest wspólnym rozkładem prawdopodobieństwa $X$ i $Y$,
- $p(x)$ i $p(y)$ są marginalnymi rozkładami prawdopodobieństwa dla $X$ i $Y$.

Informacja wzajemna jest dodatnia i osiąga wartość 0 tylko wtedy, gdy zmienne są niezależne, co oznacza, że nie mają żadnego wspólnego wpływu na siebie nawzajem.

##### Wizualizacja i interpretacja

Załóżmy, że mamy dwa zbiory danych, jeden z wyraźnym związkiem między cechą a zmienną docelową, a drugi bez związku. Poniższe wykresy pokazują te zależności:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score

# Przykład 1: wyraźny związek liniowy
np.random.seed(0)
X = np.linspace(0, 10, 100)
Y = 2 * X + np.random.normal(0, 1, 100)

# Przykład 2: brak związku
Y2 = np.random.normal(0, 1, 100)

# Wykresy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X, Y, label='Wyraźny związek')
plt.title('Wyraźny związek')
plt.xlabel('X')
plt.ylabel('Y')

plt.subplot(1, 2, 2)
plt.scatter(X, Y2, label='Brak związku', color='orange')
plt.title('Brak związku')
plt.xlabel('X')
plt.ylabel('Y')

plt.show()

# Obliczanie informacji wzajemnej
mi1 = mutual_info_score(None, np.digitize(Y, bins=10))
mi2 = mutual_info_score(None, np.digitize(Y2, bins=10))

print(f"Informacja wzajemna (związek liniowy): {mi1}")
print(f"Informacja wzajemna (brak związku): {mi2}")
```

**Wynik:**
- Informacja wzajemna jest znacznie wyższa w przypadku pierwszego przykładu, co potwierdza, że istnieje wyraźny związek między $X$ a $Y$.
- W drugim przykładzie, gdzie brak jest związku, informacja wzajemna jest bliska zeru.

##### Przykładowe zastosowanie w selekcji cech

W praktyce informacja wzajemna może być używana do selekcji cech, które są najbardziej informacyjne względem zmiennej docelowej. Na przykład, w klasyfikacji można wybrać te cechy, które mają najwyższą wartość informacji wzajemnej.

```python
from sklearn.feature_selection import mutual_info_classif

# Załóżmy, że X to macierz cech, a y to wektor etykiet
# Obliczanie informacji wzajemnej dla każdej cechy
mi_scores = mutual_info_classif(X, y)

# Wyświetlanie wyników
for i, score in enumerate(mi_scores):
    print(f"Cechy {i}: Informacja wzajemna = {score}")
```

#### Waga cech w modelach liniowych

W modelach liniowych, takich jak regresja liniowa, regresja logistyczna czy maszyny wektorów nośnych (SVM) z liniową funkcją jądrową, wagi cech odgrywają kluczową rolę w interpretacji modelu. Wartość wagi $ w_i $ w modelu liniowym wskazuje, jak silnie dana cecha wpływa na zmienną docelową.

##### Wzór na funkcję liniową

W przypadku regresji liniowej model predykcyjny jest wyrażany jako:

$$
\hat{y} = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_n x_n
$$

Gdzie:
- $\hat{y}$ to przewidywana wartość,
- $w_0$ to wyraz wolny (bias),
- $w_1, w_2, \dots, w_n$ to wagi (współczynniki) cech $x_1, x_2, \dots, x_n$.

Każda waga $w_i$ odzwierciedla zmianę w $\hat{y}$ przy jednostkowej zmianie $x_i$, zakładając, że wszystkie pozostałe cechy pozostają stałe.

##### Interpretacja wag

- **Dodatnia waga**: Wskazuje, że wzrost wartości cechy zwiększa wartość przewidywaną.
- **Ujemna waga**: Wskazuje, że wzrost wartości cechy zmniejsza wartość przewidywaną.
- **Waga bliska zeru**: Sugeruje, że cecha ma niewielki lub żaden wpływ na wartość przewidywaną.

##### Przykładowa wizualizacja

Poniższy przykład pokazuje, jak obliczyć i zwizualizować wagi cech w regresji liniowej:

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Przykładowe dane
X = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])
y = np.array([6, 10, 14, 18])

# Tworzenie modelu regresji liniowej
model = LinearRegression()
model.fit(X, y)

# Wagi cech
print(f"Wagi cech: {model.coef_}")
print(f"Wyraz wolny (bias): {model.intercept_}")

# Wizualizacja wag
plt.bar(range(len(model.coef_)), model.coef_)
plt.xlabel('Indeks cechy')
plt.ylabel('Wartość wagi')
plt.title('Wagi cech w regresji liniowej')
plt.show()
```

**Wynik:**
- Wagi pokazują, które cechy mają największy wpływ na zmienną docelową.
- Wartości wag można łatwo zinterpretować, co czyni modele liniowe bardzo intuicyjnymi.

##### Regularizacja w modelach liniowych

W modelach liniowych często stosuje się techniki regularizacji, takie jak Ridge (L2) i LASSO (L1), które wprowadzają kary za złożoność modelu. Regularizacja pomaga w radzeniu sobie z nadmiernym dopasowaniem (overfitting) oraz może prowadzić do eliminacji nieistotnych cech.

1. **Ridge (L2)**

W Ridge regresji dodaje się karę za kwadrat wag:

$$
\text{Funkcja straty} = \text{RSS} + \lambda \sum_{i=1}^{n} w_i^2
$$

Przykładowy kod:

```python
from sklearn.linear_model import Ridge

# Tworzenie modelu Ridge
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X, y)

# Wagi cech
print(f"Wagi cech (Ridge): {ridge_model.coef_}")
```

2. **LASSO (L1)**

W LASSO regresji dodaje się karę za sumę wartości bezwzględnych wag:

$$
\text{Funkcja straty} = \text{RSS} + \lambda \sum_{i=1}^{n} |w_i|
$$



Przykładowy kod:

```python
from sklearn.linear_model import Lasso

# Tworzenie modelu LASSO
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X, y)

# Wagi cech
print(f"Wagi cech (LASSO): {lasso_model.coef_}")
```

##### Zastosowanie w praktyce

W praktyce wagi cech w modelach liniowych są często wykorzystywane do interpretacji wyników modelu oraz do selekcji najważniejszych cech. Regularizacja może być stosowana, aby uprościć model, co jest szczególnie przydatne w przypadku dużych zbiorów danych z wieloma cechami.

#### Znaczenie cech w modelach nieliniowych

W modelach nieliniowych, takich jak drzewa decyzyjne, lasy losowe, gradient boosting czy sieci neuronowe, znaczenie cech jest bardziej skomplikowane do oceny niż w modelach liniowych. Modele te mogą uchwycić złożone, nieliniowe zależności między cechami a zmienną docelową, co często czyni je bardziej potężnymi, ale trudniejszymi do interpretacji.

##### Drzewa decyzyjne i lasy losowe

W modelach opartych na drzewach (np. drzewa decyzyjne i lasy losowe), znaczenie cech jest zwykle oceniane na podstawie tego, jak często cecha jest używana do podziału węzłów drzewa oraz jak bardzo poprawia czystość (ang. *purity*) danych na każdym z tych podziałów.

1. **Indeks Gini** (stosowany w klasyfikacji) - Miara nieczystości węzła, używana do oceny podziałów w drzewie decyzyjnym:

$$
G = \sum_{i=1}^{k} p_i (1 - p_i)
$$

Gdzie:
- $p_i$ to udział klasy $i$ w węźle.

2. **Redukcja nieczystości** - Znaczenie cechy w drzewach decyzyjnych mierzy się poprzez średnią redukcję nieczystości, którą wprowadza dana cecha w całym drzewie.

##### Przykład w Pythonie: Lasy losowe

Poniżej przedstawiono przykład obliczania znaczenia cech za pomocą lasów losowych:

```python
from sklearn.ensemble import RandomForestClassifier

# Załóżmy, że mamy macierz cech X i etykiety y
# Tworzenie modelu lasów losowych
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Znaczenie cech
importance = rf_model.feature_importances_
indices = np.argsort(importance)[::-1]

# Wyświetlanie rang cech
for i in range(X.shape[1]):
    print(f"Cechy {indices[i]}: Znaczenie = {importance[indices[i]]}")

# Wizualizacja
plt.bar(range(X.shape[1]), importance[indices])
plt.xlabel('Indeks cechy')
plt.ylabel('Znaczenie cechy')
plt.title('Znaczenie cech w lasach losowych')
plt.show()
```

**Wynik:**
- Wynikowe znaczenia cech pokazują, które cechy były najczęściej używane do podziałów w drzewach, co pozwala zidentyfikować kluczowe cechy.

##### Gradient Boosting

Gradient boosting to technika uczenia zespołowego, która buduje model poprzez dodawanie słabych uczących (np. drzew decyzyjnych), które są iteracyjnie dopasowywane do reszt z poprzednich modeli. W gradient boosting, podobnie jak w lasach losowych, znaczenie cech jest obliczane na podstawie ich wpływu na redukcję nieczystości w drzewach.

Przykład w Pythonie:

```python
from sklearn.ensemble import GradientBoostingClassifier

# Tworzenie modelu gradient boosting
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X, y)

# Znaczenie cech
importance = gb_model.feature_importances_

# Wizualizacja
plt.bar(range(X.shape[1]), importance[indices])
plt.xlabel('Indeks cechy')
plt.ylabel('Znaczenie cechy')
plt.title('Znaczenie cech w gradient boosting')
plt.show()
```

##### Sieci neuronowe

W sieciach neuronowych znaczenie cech jest bardziej złożone do oceny, ponieważ sieci neuronowe przekształcają dane wejściowe w skomplikowany sposób przez wiele warstw. Niemniej jednak istnieją techniki, takie jak analiza wsteczna (ang. *backpropagation*) i tzw. "DeepLIFT", które mogą być używane do oceny wpływu cech na wynik modelu.

Przykład w Pythonie (używając prostej sieci neuronowej):

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Tworzenie prostej sieci neuronowej
model = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Kompilowanie modelu
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Trenowanie modelu
model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

# Wyświetlanie wag pierwszej warstwy
weights, biases = model.layers[0].get_weights()
print("Wagi pierwszej warstwy:", weights)
```

**Wynik:**
- Wagi pierwszej warstwy w sieci neuronowej pokazują, jak silnie każda cecha jest połączona z neuronami w pierwszej ukrytej warstwie.

##### Znaczenie cech za pomocą SHAP (SHapley Additive exPlanations)

SHAP to zaawansowana metoda wyjaśnialności modelu, która przypisuje każdej cechie wartość opisującą jej wpływ na przewidywanie.

```python
import shap

# Tworzenie obiektu SHAP
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X)

# Wizualizacja SHAP dla pierwszej próbki
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1][0], X.iloc[0,:])
```

**Wynik:**
- SHAP wartości są przedstawiane jako wkład każdej cechy do końcowej predykcji, co pozwala na lepszą interpretację złożonych modeli nieliniowych.

---

© 2024 Marian Witkowski - wszelkie prawa zastrzeżone
