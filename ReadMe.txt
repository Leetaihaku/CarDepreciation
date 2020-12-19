프로젝트 진행과젱에서 찾은 K-means Modeling TIP
 - 튜플 텐서 >> 텐서 작업 시, torch.stack(target) 활용할 것
 - 파이토치 텐서 간 비교함수
    1. torch.eq() => 구성 원소 간 비교 및 텐서 반환(Tensor(T/F))
    2. torch.equal() => 텐서 자체 비교 및 부울 변수 반환(T/F)
 - 초기 값 위치에 따라 군집화 결과가 다를 수 있음(학습결과 다양화)
 - 군집 간 크기나 밀도가 다르거나, 분포가 독특한 경우 결과가 다를 수 있음(학습 부진)
 - torch.scatter 함수의 이해
    => https//aigong.tistory.com/35
 - list to tensor
    => 리스트 구조 정리 후, 마지막에 대괄호 없이 torch.tensor()로 덮기
 - cuda tensor to tensor(cpu)
    => .detach().cpu().clone().numpy()

 => 다회 수행 후, 주 결과(자주 산출)를 기준으로 채택(Majority voting)