# 1C Федоренко Екатерина. Задача № 5

Это решение 5й задачи из отбора на 1с. 

Идея работы алгоритма: в начале мы пытаемя понять, какая стратегия для нас наиболее выгодная: 
1) зажигать костер каждый раз когда не знаем куда идти и двигаться верно(не в стены)
2) тыкаться лбом в стены(проверять можем ли пройти)

Самая максимальная цена 1го действия: C + 2 * B + A
Самая максимальная цена 2го действия 3 * B + 4 * A

Сравниваем что из этого меньше и идем по этому пути.

Будем хранить несколько сетов
1) сет с инфой про клетки о которых у нас есть данные(видели их с помощью костра)
2) сет с инфой про все увиденные стены
3) сет с инфой про все проходы
4) сет с уже посещенными клетками(изначально в нем только начальная)

## Путь №1

Будем просто зажигать костер каждый раз, когда у нас не будет инфы про соседние клетки и запоминать те клетки по нужным сетам, которые видим с помощью луча
И так будем продвигаться, постепенно выстраивая граф(этакий бфс но с постепенным появлением самого графа)

## Путь №2
Каждый раз будем смотреть на клетки которые находятся вокруг и которые мы не посещали и тыкаться в них, поворачиваться и снова тыкаться.

### Окончание работы алгоритма.

Для этого должны выполниться 2 условия. 

1) Все клетки с проходами, которые были увидены посещены
2) Вокруг нашей начальной точки есть цикл из стен

Понятно, как проверить первое условие

Второе проверяем с помощью нахождения цикла в графе + проверкой принадлежности точки к полигону(образованному точками цикла)
