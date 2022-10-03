# GPU kernel 간의 성능저하 확인

Codes in mainfolder
</br>
</br>
### base_info.h
GPU의 SM을 결정하기 위해 grid size와 block size가 수정 가능한 header file
</br>
</br>
### get_random.cu
Random seed와 number set, matrix를 받기 위한 함수들
</br>
</br>
### high_core.cu
Core intensive한 연산을 위해 dependent한 사칙연산을 반복한 code
</br>
</br>
### matrix_cal.cu
SM내에 행렬 곱을 연산하기 위한 함수들
d_mm_normal은 일반적인 연산
d_mm_shared는 shared memory를 활용한 연산
</br>
</br>
### mm_normal.cu && mm_shared.cu
각각의 함수를 호출해서 행렬 곱에 걸리는 시간을 측정하는 codes
</br>
</br>
### transpose.cu
행렬을 transpose 하는 code
