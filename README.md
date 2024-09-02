# Wybór cech w modelu uczenia maszynowego

### Autor: Marian Witkowski

---

**Spis Treści**

1. **<a href='01-wstep.md'>Wstęp</a>**
   - Definicja uczenia maszynowego
   - Rola cech w modelach uczenia maszynowego
   - Znaczenie wyboru odpowiednich cech

2. **<a href='02-przeglad.md'>Przegląd Metod Wyboru Cech</a>**
   - Selekcja cech oparta na filtrach
     - Statystyczne testy istotności
     - Metody miar podobieństwa (np. korelacja)
   - Selekcja cech oparta na osadzaniu (embedded)
     - Drzewa decyzyjne i lasy losowe
     - Regularizacja (LASSO, Ridge)
   - Selekcja cech oparta na przeszukiwaniu (wrapper)
     - Metody do przeszukiwania przestrzeni cech (np. RFE)
     - Cross-validation w wyborze cech

3. **<a href='03-kryteria.md'>Kryteria Oceny Cech</a>**
   - Informacja wzajemna
   - Waga cech w modelach liniowych
   - Znaczenie cech w modelach nieliniowych

4. **<a href='04-wybor.md'>Proces Wyboru Cech</a>**
   - Preprocessing danych
     - Czyszczenie danych
     - Normalizacja i standaryzacja
   - Eksploracyjna analiza danych (EDA)
     - Analiza korelacji
     - Wykresy rozrzutu
   - Implementacja metod selekcji cech
     - Przykłady zastosowania różnych metod
     - Porównanie wyników różnych metod

5. **<a href='05-zastosowanie.md'>Zastosowanie Wyboru Cech w Różnych Typach Modeli</a>**
   - Modele liniowe (np. regresja liniowa)
   - Modele drzewiaste (np. drzewa decyzyjne, lasy losowe)
   - Modele oparte na kernelach (np. SVM)
   - Modele sieci neuronowych

6. **<a href='06-przypadki-uzycia.md'>Przypadki Użycia</a>**
   - Studia przypadków z rzeczywistych projektów
   - Porównanie efektywności modeli z różnymi zestawami cech
   - Analiza wyników i wnioski

7. **<a href='07-wyzwania.md'>Wyzwania i Problemy w Wyborze Cech</a>**
   - Przekleństwo wymiarowości
   - Problem nadmiernego dopasowania (overfitting)
   - Radzenie sobie z cechami redundantnymi i nieistotnymi

8. **<a href='08-narzedzia.md'>Narzędzia i Biblioteki do Wyboru Cech</a>**
   - Scikit-learn
   - Feature-engine
   - Boruta
   - Inne popularne biblioteki

9. **<a href='09-trendy.md'>Przyszłość i Trendy w Wyborze Cech</a>**
   - Automatyzacja wyboru cech
   - Wykorzystanie głębokiego uczenia do automatycznego odkrywania cech
   - Rola wyjaśnialności modelu w wyborze cech

10. **<a href='10-podsumowanie.md'>Podsumowanie</a>**
    - Kluczowe wnioski
    - Rekomendacje dotyczące najlepszych praktyk w wyborze cech
    - Przyszłe kierunki badań

11. **<a href='11-bibliografia.md'>Bibliografia</a>**
    - Spis cytowanych publikacji i źródeł

---