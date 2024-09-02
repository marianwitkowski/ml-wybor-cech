## 8. **Narzędzia i Biblioteki do Wyboru Cech**

Wybór odpowiednich cech jest jednym z kluczowych etapów procesu budowy modeli uczenia maszynowego. Na szczęście, istnieje wiele narzędzi i bibliotek, które mogą znacznie ułatwić i przyspieszyć ten proces. W tej sekcji omówimy kilka popularnych narzędzi i bibliotek do selekcji cech, takich jak Scikit-learn, Feature-engine, Boruta oraz inne. Każda z tych bibliotek oferuje różne metody selekcji cech, które mogą być używane w zależności od specyfiki problemu i rodzaju danych.

#### Scikit-learn

##### Wprowadzenie

Scikit-learn to jedna z najpopularniejszych bibliotek do uczenia maszynowego w Pythonie. Oferuje szeroki wachlarz narzędzi do przetwarzania danych, modelowania, selekcji cech i oceny modeli. W kontekście selekcji cech, Scikit-learn dostarcza kilka metod, które są łatwe w użyciu i zintegrowane z innymi narzędziami biblioteki.

##### Metody selekcji cech w Scikit-learn

1. **SelectKBest:**

   SelectKBest to jedna z najprostszych metod selekcji cech, która wybiera k najlepszych cech na podstawie określonego testu statystycznego, takiego jak chi-kwadrat (dla danych kategorialnych) lub ANOVA F-value (dla danych ciągłych).

   Przykładowy kod:

   ```python
   from sklearn.feature_selection import SelectKBest, chi2

   # Wybór najlepszych cech za pomocą testu chi-kwadrat
   selector = SelectKBest(chi2, k=5)
   X_new = selector.fit_transform(X, y)

   print("Wybrane cechy:\n", X.columns[selector.get_support()])
   ```

   Wzór na test chi-kwadrat:

   $\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}$

   Gdzie:
   - $O_i$ to obserwowana liczba zdarzeń w kategorii $i$,
   - $E_i$ to oczekiwana liczba zdarzeń w kategorii $i$.

2. **Recursive Feature Elimination (RFE):**

   RFE to iteracyjna metoda selekcji cech, która trenowuje model na pełnym zestawie cech, a następnie usuwa najmniej istotne cechy, aż do osiągnięcia pożądanego podzbioru cech.

   Przykładowy kod:

   ```python
   from sklearn.feature_selection import RFE
   from sklearn.linear_model import LogisticRegression

   # Model bazowy
   model = LogisticRegression()

   # RFE
   rfe = RFE(model, n_features_to_select=5)
   rfe.fit(X, y)

   print("Wybrane cechy:\n", X.columns[rfe.support_])
   ```

   Wzór na funkcję kosztu w regresji logistycznej:

   $J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right]$

   Gdzie:
   - $m$ to liczba próbek,
   - $y^{(i)}$ to rzeczywista etykieta,
   - $h_\theta(x^{(i)})$ to przewidywana etykieta.

3. **Feature Importance:**

   W modelach takich jak drzewa decyzyjne czy lasy losowe, Scikit-learn umożliwia ocenę ważności cech, co pozwala na wybór tych najbardziej istotnych dla predykcji.

   Przykładowy kod:

   ```python
   from sklearn.ensemble import RandomForestClassifier

   # Model lasów losowych
   rf = RandomForestClassifier(n_estimators=100)
   rf.fit(X, y)

   # Znaczenie cech
   importances = rf.feature_importances_
   indices = np.argsort(importances)[::-1]

   # Wyświetlanie wyników
   for i in range(X.shape[1]):
       print(f"Cechy {X.columns[indices[i]]}: Ważność = {importances[indices[i]]}")

   # Wizualizacja wyników
   plt.figure(figsize=(10, 6))
   plt.title("Znaczenie cech w lasach losowych")
   plt.bar(range(X.shape[1]), importances[indices], align="center")
   plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
   plt.show()
   ```

   Wzór na obliczanie ważności cech:

   $\text{Ważność cechy} = \frac{1}{T} \sum_{t=1}^{T} I_t$

   Gdzie:
   - $T$ to liczba drzew w modelu,
   - $I_t$ to wartość informacji zysku dla cechy w drzewie $t$.

##### Przykład zastosowania

Załóżmy, że pracujemy nad projektem klasyfikacji wniosków kredytowych. W takim przypadku możemy użyć RFE, aby zredukować liczbę cech i poprawić wydajność modelu, jednocześnie utrzymując wysoką dokładność predykcji.

#### Feature-engine

##### Wprowadzenie

Feature-engine to biblioteka Python, która specjalizuje się w przetwarzaniu cech i selekcji cech. Jest zintegrowana z Scikit-learn, co ułatwia jej stosowanie w istniejących pipeline'ach. Feature-engine oferuje narzędzia do inżynierii cech, takich jak kodowanie zmiennych kategorialnych, obsługa wartości odstających, tworzenie cech na podstawie cech istniejących oraz selekcję cech.

##### Metody selekcji cech w Feature-engine

1. **DropCorrelatedFeatures:**

   DropCorrelatedFeatures usuwa cechy, które są silnie skorelowane z innymi cechami, co pozwala na redukcję redundancji w zbiorze danych.

   Przykładowy kod:

   ```python
   from feature_engine.selection import DropCorrelatedFeatures

   # Dropowanie silnie skorelowanych cech
   drop_corr = DropCorrelatedFeatures(threshold=0.8)
   X_reduced = drop_corr.fit_transform(X)

   print("Wybrane cechy po usunięciu korelacji:\n", X_reduced.columns)
   ```

   Wzór na współczynnik korelacji Pearsona:

   $r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}$

   Gdzie:
   - $x_i$ i $y_i$ to wartości cech,
   - $\bar{x}$ i $\bar{y}$ to średnie wartości cech.

2. **SmartCorrelatedSelection:**

   SmartCorrelatedSelection idzie o krok dalej i usuwa tylko te cechy, które są mniej istotne w przypadku wysokiej korelacji, na podstawie ważności cech wyznaczonych przez model.

   Przykładowy kod:

   ```python
   from feature_engine.selection import SmartCorrelatedSelection
   from sklearn.ensemble import RandomForestClassifier

   # SmartCorrelatedSelection
   rf = RandomForestClassifier()
   smart_corr = SmartCorrelatedSelection(estimator=rf, threshold=0.8)
   X_smart = smart_corr.fit_transform(X, y)

   print("Wybrane cechy po SmartCorrelatedSelection:\n", X_smart.columns)
   ```

3. **RecursiveFeatureElimination:**

   Feature-engine oferuje również swoją implementację RFE, która integruje się z innymi narzędziami tej biblioteki.

   Przykładowy kod:

   ```python
   from feature_engine.selection import RecursiveFeatureElimination
   from sklearn.ensemble import RandomForestClassifier

   # Recursive Feature Elimination
   rfe = RecursiveFeatureElimination(estimator=RandomForestClassifier(), scoring='roc_auc', threshold=0.01)
   X_rfe = rfe.fit_transform(X, y)

   print("Wybrane cechy po RFE:\n", X_rfe.columns)
   ```

##### Przykład zastosowania

Załóżmy, że pracujemy nad projektem dotyczącym przewidywania churnu klientów w telekomunikacji. Feature-engine może być użyty do usunięcia cech silnie skorelowanych lub do zastosowania SmartCorrelatedSelection, aby skupić się na tych cechach, które są najważniejsze z punktu widzenia modelu.

#### Boruta

##### Wprowadzenie

Boruta to algorytm selekcji cech oparty na metodzie lasów losowych, który jest zaprojektowany, aby zidentyfikować wszystkie cechy istotne dla przewidywań. Boruta działa, tworząc kopie losowych cech (shadow features) i sprawdzając, czy rzeczywiste cechy przewyższają te losowe, co oznacza, że są istotne.

##### Jak działa Boruta?

Boruta iteracyjnie trenuje lasy losowe na zbiorze danych, gdzie każda cecha rzeczywista jest porównywana z cechą losową (shadow). Jeśli cecha rzeczywista jest bardziej istotna niż shadow, jest uznawana za istotną; w przeciwnym razie jest odrzucana.

Przykładowy kod w Pythonie:

```python
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier

# Model bazowy
rf = RandomForestClassifier(n_estimators=100)

# Boruta
boruta_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=42)
boruta_selector.fit(X.values, y.values)

# Wynik selekcji
selected_features = X.columns[boruta_selector.support_].tolist()
print("Wybrane cechy przez Boruta:\n", selected_features)
```

Wzór na funkcję kosztu lasów losowych:

$\text{Impurity} = \sum_{t=1}^{T} \frac{n_t}{N} I_t$

Gdzie:
- $T$ to liczba drzew w modelu,
- $n_t$ to liczba próbek w węźle $t$,
- $I_t$ to nieczystość węzła.

##### Przykład zastosowania

W projekcie dotyczącym analizy genetycznej, gdzie liczba cech może być ogromna, Boruta może być użyty do zidentyfikowania wszystkich istotnych genów związanych z konkretnym schorzeniem, eliminując przy tym te, które nie wnoszą wartości do modelu.

#### Inne popularne biblioteki

##### LIME (Local Interpretable Model-agnostic Explanations)

LIME to biblioteka, która służy do wyjaśniania predykcji modeli uczenia maszynowego poprzez generowanie lokalnych wyjaśnień dla indywidualnych predykcji. Choć nie jest to bezpośrednio narzędzie do selekcji cech, LIME może pomóc zrozumieć, które cechy są najważniejsze dla konkretnych predykcji, co może informować proces selekcji cech.

Przykładowy kod w Pythonie:

```python
import lime
import lime.lime_tabular

# Tworzenie wyjaśnienia dla modelu
explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, feature_names=X_train.columns, class_names=['0', '1'], verbose=True, mode='classification')
exp = explainer.explain_instance(X_test.iloc[0].values, model.predict_proba, num_features=5)

# Wizualizacja wyjaśnienia
exp.show_in_notebook(show_all=False)
```

##### SHAP (SHapley Additive exPlanations)

SHAP to inna biblioteka do wyjaśniania predykcji, która wykorzystuje wartości Shapleya z teorii gier do oceny ważności cech. SHAP dostarcza bardziej globalnej perspektywy na znaczenie cech w całym zbiorze danych, co może wspomagać selekcję cech.

Przykładowy kod w Pythonie:

```python
import shap

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Tworzenie wyjaśnienia za pomocą SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Wizualizacja SHAP dla pierwszej próbki
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1][0], X_test.iloc[0,:])
```

##### MLXtend

MLXtend to biblioteka, która dostarcza wiele przydatnych narzędzi do rozszerzenia możliwości Scikit-learn, w tym narzędzia do selekcji cech, takie jak Sequential Feature Selector (SFS), który pozwala na iteracyjne dodawanie lub usuwanie cech, aż do osiągnięcia optymalnego zestawu.

Przykładowy kod w Pythonie:

```python
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.neighbors import KNeighborsClassifier

# Sequential Feature Selection
sfs = SFS(KNeighborsClassifier(n_neighbors=3), k_features=5, forward=True, floating=False, scoring='accuracy', cv=5)
sfs = sfs.fit(X, y)

print("Wybrane cechy:\n", sfs.k_feature_names_)
```

##### Przykład zastosowania

Załóżmy, że budujemy model do przewidywania wyników egzaminów na podstawie danych demograficznych, historii edukacji i wyników wcześniejszych testów. SHAP może być użyty do identyfikacji cech, które mają największy wpływ na predykcje, co może informować o tym, które cechy należy zachować w modelu.

---

© 2024 Marian Witkowski - wszelkie prawa zastrzeżone
