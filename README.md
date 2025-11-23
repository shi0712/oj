# Online Judge

Windows平台的简易OJ系统，支持SPJ（Special Judge）。

## 功能

- 支持多个G++版本 (C++14/17/20/23)
- 支持SPJ (testlib.h)
- 并发评测
- Web前端界面

## 安装

1. 安装Python依赖:
```bash
cd backend
pip install -r requirements.txt
```

2. 解压compilers文件夹中的compilers.zip

3. 配置编译器路径:
   - 修改 `backend/config.py` 中的 `COMPILERS` 配置
   - 根据实际安装路径修改各版本g++的path
   - 也可以通过Web界面的"编译器配置"页面设置

## 运行

```bash
cd backend
python main.py
```

访问 http://localhost:8000

## API接口

### 题目管理

- `POST /api/problems` - 上传题目
  - `problem_id`: 题目ID
  - `title`: 标题
  - `time_limit`: 时间限制(ms)
  - `memory_limit`: 空间限制(MB)
  - `testcases`: ZIP文件(包含.in和.out)
  - `checker`: checker.cpp (可选)

- `GET /api/problems` - 获取题目列表
- `GET /api/problems/{id}` - 获取题目详情
- `DELETE /api/problems/{id}` - 删除题目

### 提交评测

- `POST /api/submit` - 提交代码
  - `problem_id`: 题目ID
  - `code`: 源代码
  - `language`: c++14/c++17/c++20/c++23

- `GET /api/submissions/{id}` - 获取评测结果
- `GET /api/submissions` - 获取提交列表

### Hack测试

- `POST /api/hack` - 使用自定义输入测试代码
  - `problem_id`: 题目ID
  - `code`: 待测试的用户代码
  - `language`: c++14/c++17/c++20/c++23
  - `input_data`: 自定义输入数据
  - `std_code`: 标准程序代码（可选，用于生成期望输出）

返回:
```json
{
  "status": "Accepted",      // 评测结果
  "time_used": 15,           // 运行时间(ms)
  "output": "User Output:\n..." // 用户输出和对比信息
}
```

可能的status值:
- `Accepted` - 输出正确
- `Wrong Answer` - 输出错误（与std或checker判定不符）
- `Time Limit Exceeded` - 超时
- `Runtime Error` - 运行时错误
- `Compile Error` - 用户代码编译错误
- `System Error` - 系统错误（std编译失败、checker错误等）

使用示例:
```bash
# 只运行代码，查看输出
curl -X POST http://localhost:8000/api/hack \
  -F "problem_id=A" \
  -F "language=c++17" \
  -F "input_data=1 2" \
  -F "code=#include<iostream>
int main(){int a,b;std::cin>>a>>b;std::cout<<a+b;}"

# 提供std代码进行对比
curl -X POST http://localhost:8000/api/hack \
  -F "problem_id=A" \
  -F "language=c++17" \
  -F "input_data=1 2" \
  -F "code=..." \
  -F "std_code=..."
```

工作流程:
1. 编译用户代码（失败返回CE）
2. 编译std代码（如果提供，失败返回SE）
3. 编译checker（如果题目有SPJ）
4. 运行用户代码（TLE/RE则直接返回）
5. 运行std代码生成期望输出
6. 使用checker或直接比较判断结果

### 配置

- `GET /api/languages` - 获取支持的语言
- `POST /api/config/compiler` - 设置编译器路径

## 评测状态

- `AC` - Accepted
- `WA` - Wrong Answer
- `TLE` - Time Limit Exceeded
- `MLE` - Memory Limit Exceeded
- `RE` - Runtime Error
- `CE` - Compile Error
- `SE` - System Error

## 测试数据格式

ZIP文件包含:
- `1.in`, `1.out`
- `2.in`, `2.out`
- ...

## SPJ Checker

使用testlib.h编写checker.cpp:

```cpp
#include "testlib.h"

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    // inf: 输入文件
    // ouf: 用户输出
    // ans: 标准答案

    int expected = ans.readInt();
    int actual = ouf.readInt();

    if (expected == actual) {
        quitf(_ok, "Correct");
    } else {
        quitf(_wa, "Expected %d, got %d", expected, actual);
    }
}
```

