## Solo vs Debate/Single (with Tools vs without Tools)

### 36-37:

otoal = A @ x + B @ x, And the descriptiton allow the instep result to be approx, we may require the total error within 10%. 

The kernel quantanize the A and B's wieghts first then multiply and add.

36 is make the error of the A @ x, B @ x add in one direction. but 37 is offset.

solo see the single error is larger then reject but as description as long as total is good it will be correct

### 38-39
Attention calculation

### 40–41
Kernel is status = A @ status + current_input. Use the high precision final status as the reference. allow 0.2% error. Description allow approc mid step result

Kernel is status = A @ status + current_input. Every step of update, convert status -> fp16

so 40 and 41 is just input different? 
Then this kernel should be accept or reject. This is not decided by input you know ?

### 44–45
LayerNorm. The ref variance is ((x - mu) ** 2).mean(), but the kernel write it to (x ** 2).mean() - mu**2. They are mathematically equal. 

so what is the difference between 44 and 45

so this kernel should be rejected or accepted? This decision should not be decidec by inputs

### 50–51
totally i do not understand

### 58–59

FP32 log-determinant. i do not understand 

### 60–61
‖a‖² + ‖q‖² − 2a·q instead the kernel will calculate convert to ‖a−q‖². Mathematically equal. UNder fp32 several big number delete will throw away the small distance info?

**想测什么：** 既不能因距离公式数学等价就直接接受，也不能因中间距离存在相消风险就直接拒绝；需要追踪到最终预测。

Is this a manipulation of the input?
