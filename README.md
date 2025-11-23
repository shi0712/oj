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

