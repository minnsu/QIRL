# Environment 클래스에 대해

1. chart_data, price_idx, init_cash를 필수로 전달한다.
    - chart_data.shape = (n_days, 1day_data)
    - price_idx는 1day_data 중 몇번째에 위치하는지를 나타낸다.

2. step 함수는 