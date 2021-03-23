# 베이지안-최적화 패키지

**파라미터 튜닝 자동화 패키지**

- 자료 구성
    1. [basic-tour, 기초 수준]
    2. -

- 참고자료
    - [github/fmfn, 공식문서](https://l.facebook.com/l.php?u=https%3A%2F%2Fgithub.com%2Ffmfn%2FBayesianOptimization%3Ffbclid%3DIwAR2u2NbdzAtiYab3C6T_5-W67n3w0-Ca_2QVAQFeVgXcDxwnbIUxDWN-pX0&h=AT21iPftbKyIz1UGBcK8aQafzsCrNUl5m587LDO-eZeISdAzjFhEwCIRgicbO-DUoYpwWZHQG4TxZqW3Vt6eEIpCaf4nFuMBFkEgymFj1O6QvXYs4PJ5esc2XTHZt8LMTsgdFgB6bg&__tn__=H-R&c[0]=AT3Zq8kOIvURZdAEFmJCGbsYy9FCsrRiW8KZkeV4XN7dPIajDtQbOqAQ-NSZNhO_TqP8Wlxq6z1JuLzb23gG_FMml1XRX5aWI_qZ8F-i5nbEX1e0078n2Rxyj0QZ7nh4Lpt99uZphn3Nn_hKJSvDfC_2kemGAptKLz8)
    - [주의 사항](https://www.youtube.com/watch?v=DGJTEBt0d-s)
    
- 사용하면 안되는 상황 요약
    - f is black box functionwith no closed form nor gradients
    - f is expensive to evaluate
    - you may have only noisy obs of f