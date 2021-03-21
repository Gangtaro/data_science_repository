# data_science_repository
this is my repository of data science works

## jupyter notebook 가상환경 연결하기

### 1. 아나콘다 가상환경 설치하기
---

#### 가상환경 이름 및 원하는 버전의 파이썬으로 설치
```zsh
conda create -n [virtual env. name] python==[version which you want]
```
- [virtual env. name] : 만들고 싶은 가상환경의 이름 
- python== : (입력하지 않으면 최신 버전)
    - [version which you want] : 원하는 버전의 파이썬 ex) python==3.7.10
    
#### 삭제
```zsh
conda remove [virtual env. name]
```

#### conda에 설치되어있는 python list 확인
```zsh
conda search python
```

### 2. virtual env.(가상환경) 활성화
---
#### 활성화 가능한 가상환경 리스트 확인
```zsh
conda env list
```
#### 활성화
```zsh
conda activate [virtual env. name]
```
#### 비활성화
```zsh
deactivate []
```

### 3. 가상환경에 Jupyter notebook 연결 
---
**주피터 노트북은 이미 설치되어 있다는 전제 하**
```zsh
pip install ipykernel
```

```zsh
python -m ipykernel install --user --name [virtual env. name] --display-name "[displayKenrelName]"
```
- [displayKenrelName] : jupyter notebook 내에서 표시할 커널 이름




