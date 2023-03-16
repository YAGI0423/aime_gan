### 이 저장소(Repository)는 「파이토치(Pytorch)를 위한 논리 게이트(Logic Gate) 데이터셋」에 대한 내용을 다루고 있습니다.

***
작성자: YAGI<br>

최종 수정일: 2023-03-16
+ 2023.03.15: 코드 작성 완료
+ 2023.03.16: README 작성 완료
***

<br>

***
+ 프로젝트 기간: 2023-03-15 ~ 2023-03-16
***
<br>

## 프로젝트 요약
&nbsp;&nbsp;
파이토치(Pytorch)의 'Dataset' 형식으로 된 논리 게이트(Logic Gate) 데이터셋을 제공합니다. AND, OR, XOR, NOT 총 네 개의 논리 연산자 기반의 데이터셋을 사용할 수 있습니다. 기존 파이토치 Dataset과 마찬가지로 DataLoader를 이용하여 순회 가능한 객체(Iterable)를 구현할 수 있습니다.
<br><br>

## Getting Start

### Example
```python
$ python example.py

>>>
=========< XOR Dataset Sample >=========
    X         Y
[[0. 0.]]   [[0.]]
[[0. 1.]]   [[1.]]
[[1. 0.]]   [[1.]]
[[1. 1.]]   [[0.]]

========================================
```
<br>

### Get Logic Dataset
```python
from logicGateDataset.datasets import AndGate, OrGate, XorGate, NotGate

#Get AND Dataset
#input_size: 입력 x의 개수, default=2
#dataset_size: 데이터셋의 전체 크기
dataset = AndGate(input_size=2, datset_size=100)

#Get XOR Dataset
dataset = XorGate(input_size=2, datset_size=100)

#DataLoader
dataLoader = DataLoader(dataset, batch_size=4, shuffle=False)
```
***
<br><br>


## 개발 환경
**Language**: Python 3.9.12

**Library**
    + pytorch 1.12.0

<br><br>

## License
This project is licensed under the terms of the [MIT license](https://github.com/YAGI0423/logicGate_dataset/blob/main/LICENSE).