# Selenium 사용 가이드 

**시스템 정보**  
- 운영체제: macOS 11.2.3

**conda virtualenv(가상환경) 정보**
- python version == 3.7.10
- selenium version == 3.141.0

**참고자료**
- [firefox를 이용한 티케팅 자동화 튜토리얼](https://gem1n1.tistory.com/38)
- [파이썬 셀레니움을 이용한 데이터 크롤링 환경 구축](https://blog.naver.com/PostView.nhn?blogId=rlacksdid93&logNo=221971523684)
- 

## 설치 가이드 (macOS)

### For Chrome browser
#### 드라이버 다운로드
1. **크롬 브라우저 버전 확인**  
    점 세 개 > 도움말 > chrome 정보 --> 버전 확인 ex)버전 89.0.4389.90(공식 빌드) (x86_64)

2. **버전과 일치하는 크롬드라이버 다운로드**  
    [크롬드라이버 다운로드 링크](https://sites.google.com/a/chromium.org/chromedriver/downloads)

3. **크롬드라이버 올바른 위치로 이동**  
    해당파일의 올바른 경로는 다음과 같다.  
    /usr/local/bin/chromedriver

#### 셀레니움(selenium 라이브러리 설치)
```zsh
pip install selenium
```

```zsh
sudo pip install selenium
conda install -c conda-forge selenium
```
