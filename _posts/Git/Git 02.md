# Git 02



## .gitignore(git에서 관리하기 싫은 것들)

### git 포함, 배제

#### 	git에서 소스코드를 관리, github에서 소스코드 공유할 때, 반드시 필요한 코드와 민감성 정보 등 포함되지 않아야 할 것들이 같은 폴더에 있을 경우, .gitignore을 사용해서 코드를 포함, 배제 할 수 있다.(기본값 : 포함)

### .gitignore 문법

```.gitignore
# 파일 명이 file.c인 파일 모두 배제
file.c

# 최상위 폴더의 file.c 만 배제
/file.c

# 모든 .c 확장자 파일
*.c

# .c 확장자 파일을 배제했지만, 포함하고 싶은 파일(배제를 배제 > 포함)
!not_ignore_file.c

# logs란 파일 또는 폴더 모두 배제
logs

# logs란 폴더와 그 내용들 모두 배제(logs 파일은 배제 X)
logs/

# logs 폴더 안의 file.c 배제
logs/file.c

# logs 폴더 바로 안, 또는 그 안에 다른 폴더들 속의 file.c
logs/**/file.c
```

