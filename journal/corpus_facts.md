==============================================================================
[tutorial/first-steps.md]  行数=428  命中=15
    12| FastAPI 有一个[官方 VS Code 扩展](https://marketplace.visualstudio.com/items?itemName=FastAPILabs.fastapi-vscode)（以及 Cursor），它提供了很多功能，包括路径操作浏览器、路径操作搜索、测试中的 C
    16| 运行实时服务器：
   254| ### 步骤 3：创建一个*路径操作* { #step-3-create-a-path-operation }
   315| #### 定义一个*路径操作装饰器* { #define-a-path-operation-decorator }
   319| `@app.get("/")` 告诉 **FastAPI** 在它下方的函数负责处理如下访问请求：
   326| `@something` 语法在 Python 中被称为「装饰器」。
   330| 装饰器接收位于其下方的函数并且用它完成一些工作。
   332| 在我们的例子中，这个装饰器告诉 **FastAPI** 位于其下方的函数对应着**路径** `/` 加上 `get` **操作**。
   334| 它是一个「**路径操作装饰器**」。
   363| ### 步骤 4：定义**路径操作函数** { #step-4-define-the-path-operation-function }

==============================================================================
[tutorial/path-params.md]  行数=251  命中=28
     1| # 路径参数 { #path-parameters }
     7| 路径参数 `item_id` 的值会作为参数 `item_id` 传递给你的函数。
    12| {"item_id":"foo"}
    15| ## 声明路径参数的类型 { #path-parameters-with-types }
    17| 使用 Python 标准类型注解，声明路径操作函数中路径参数的类型：
    21| 本例把 `item_id` 的类型声明为 `int`。
    25| 类型声明将为函数提供错误检查、代码补全等编辑器支持。
    34| {"item_id":3}
    41| **FastAPI** 通过类型声明自动进行请求的<dfn title="将来自 HTTP 请求中的字符串转换为 Python 数据类型">“解析”</dfn>。
    56| "item_id"

==============================================================================
[tutorial/query-params.md]  行数=189  命中=37
     1| # 查询参数 { #query-parameters }
     6| {* ../../docs_src/query_params/tutorial001_py310.py hl[9] *}
    13| http://127.0.0.1:8000/items/?skip=0&limit=10
    16| ...查询参数为：
    18| * `skip`：值为 `0`
    19| * `limit`：值为 `10`
    25| 所有应用于路径参数的流程也适用于查询参数：
    32| ## 默认值 { #defaults }
    34| 查询参数不是路径的固定内容，它是可选的，还支持默认值。
    36| 上例用 `skip=0` 和 `limit=10` 设定默认值。

==============================================================================
[tutorial/body.md]  行数=166  命中=35
     1| # 请求体 { #request-body }
     3| 当你需要从客户端（比如浏览器）向你的 API 发送数据时，会把它作为**请求体**发送。
     5| **请求体**是客户端发送给你的 API 的数据。**响应体**是你的 API 发送给客户端的数据。
     7| 你的 API 几乎总是需要发送**响应体**。但客户端不一定总是要发送**请求体**，有时它们只请求某个路径，可能带一些查询参数，但不会发送请求体。
     9| 使用 [Pydantic](https://pydantic.dev/docs/) 模型来声明**请求体**，能充分利用它的功能和优点。
    15| 规范中没有定义用 `GET` 请求发送请求体的行为，但 FastAPI 仍支持这种方式，只用于非常复杂/极端的用例。
    17| 由于不推荐，在使用 `GET` 时，Swagger UI 的交互式文档不会显示请求体的文档，而且中间的代理可能也不支持它。
    21| ## 导入 Pydantic 的 `BaseModel` { #import-pydantics-basemodel }
    23| 首先，你需要从 `pydantic` 中导入 `BaseModel`：
    25| {* ../../docs_src/body/tutorial001_py310.py hl[2] *}

==============================================================================
[tutorial/dependencies/index.md]  行数=250  命中=39
     1| # 依赖项 { #dependencies }
     3| **FastAPI** 提供了简单直观但功能强大的**<dfn title="也称为：组件、资源、提供者、服务、可注入项">依赖注入</dfn>**系统。
     7| ## 什么是「依赖注入」 { #what-is-dependency-injection }
     9| 在编程中，**「依赖注入」**指的是，你的代码（本文中为*路径操作函数*）声明其运行所需并要使用的东西：“依赖”。
    11| 然后，由该系统（本文中为 **FastAPI**）负责执行所有必要的逻辑，为你的代码提供这些所需的依赖（“注入”依赖）。
    26| 但这样我们就可以专注于**依赖注入**系统是如何工作的。
    28| ### 创建依赖项，或“dependable” { #create-a-dependency-or-dependable }
    30| 首先关注依赖项。
    46| 本例中的依赖项预期接收：
    64| ### 导入 `Depends` { #import-depends }

==============================================================================
[tutorial/middleware.md]  行数=95  命中=31
     1| # 中间件 { #middleware }
     3| 你可以向 **FastAPI** 应用添加中间件。
     5| “中间件”是一个函数，它会在每个特定的*路径操作*处理每个**请求**之前运行，也会在返回每个**响应**之前运行。
     7| * 它接收你的应用的每一个**请求**。
     8| * 然后它可以对这个**请求**做一些事情或者执行任何需要的代码。
     9| * 然后它将这个**请求**传递给应用程序的其他部分（某个*路径操作*）处理。
    10| * 之后它获取应用程序生成的**响应**（由某个*路径操作*产生）。
    11| * 它可以对该**响应**做一些事情或者执行任何需要的代码。
    12| * 然后它返回这个**响应**。
    16| 如果你有使用 `yield` 的依赖，依赖中的退出代码会在中间件之后运行。

==============================================================================
[tutorial/handling-errors.md]  行数=244  命中=33
     1| # 处理错误 { #handling-errors }
     3| 某些情况下，需要向使用你的 API 的客户端返回错误提示。
    18| 而 400 范围内的状态码表示客户端发生了错误。
    20| 大家都知道**「404 Not Found」**错误，还有调侃这个错误的笑话吧？
    22| ## 使用 `HTTPException` { #use-httpexception }
    24| 向客户端返回 HTTP 错误响应，可以使用 `HTTPException`。
    26| ### 导入 `HTTPException` { #import-httpexception }
    30| ### 在代码中触发 `HTTPException` { #raise-an-httpexception-in-your-code }
    32| `HTTPException` 是额外包含了和 API 有关数据的常规 Python 异常。
    36| 这也意味着，如果你在*路径操作函数*里调用的某个工具函数内部触发了 `HTTPException`，那么*路径操作函数*中后续的代码将不会继续执行，请求会立刻终止，并把 `HTTPException` 的 HTTP 错误发送给客户端。

==============================================================================
[advanced/events.md]  行数=166  命中=26
     1| # 生命周期事件 { #lifespan-events }
     4| 你可以定义在应用**启动**前执行的逻辑（代码）。这意味着在应用**开始接收请求**之前，这些代码只会被执行**一次**。
     6| 同样地，你可以定义在应用**关闭**时应执行的逻辑。在这种情况下，这段代码将在**处理可能的多次请求后**执行**一次**。
     8| 因为这段代码在应用开始接收请求**之前**执行，也会在处理可能的若干请求**之后**执行，它覆盖了整个应用程序的**生命周期**（“生命周期”这个词很重要😉）。
    26| ## Lifespan { #lifespan }
    28| 你可以使用 `FastAPI` 应用的 `lifespan` 参数和一个“上下文管理器”（稍后我将为你展示）来定义**启动**和**关闭**的逻辑。
    32| 我们使用 `yield` 创建了一个异步函数 `lifespan()` 像这样：
    36| 在这里，我们在 `yield` 之前将（虚拟的）模型函数放入机器学习模型的字典中，以此模拟加载模型的耗时**启动**操作。这段代码将在应用程序**开始处理请求之前**执行，即**启动**期间。
    38| 然后，在 `yield` 之后，我们卸载模型。这段代码将会在应用程序**完成处理请求后**执行，即在**关闭**之前。这可以释放诸如内存或 GPU 之类的资源。
    42| **关闭**事件会在你**停止**应用时发生。

==============================================================================
[tutorial/cors.md]  行数=88  命中=14
     1| # CORS（跨域资源共享） { #cors-cross-origin-resource-sharing }
     3| [CORS 或者「跨域资源共享」](https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS) 指浏览器中运行的前端拥有与后端通信的 JavaScript 代码，而后端处于与前端不同的「源」的情况。
    35| ## 使用 `CORSMiddleware` { #use-corsmiddleware }
    37| 你可以在 **FastAPI** 应用中使用 `CORSMiddleware` 来配置它。
    39| * 导入 `CORSMiddleware`。
    51| 默认情况下，这个 `CORSMiddleware` 实现所使用的默认参数较为保守，所以你需要显式地启用特定的源、方法或者 headers，以便浏览器能够在跨域上下文中使用它们。
    55| * `allow_origins` - 一个允许跨域请求的源列表。例如 `['https://example.org', 'https://www.example.org']`。你可以使用 `['*']` 允许任何源。
    56| * `allow_origin_regex` - 一个正则表达式字符串，匹配的源允许跨域请求。例如 `'https://.*\.example\.org'`。
    57| * `allow_methods` - 一个允许跨域请求的 HTTP 方法列表。默认为 `['GET']`。你可以使用 `['*']` 来允许所有标准方法。
    58| * `allow_headers` - 一个允许跨域请求的 HTTP 请求头列表。默认为 `[]`。你可以使用 `['*']` 允许所有的请求头。`Accept`、`Accept-Language`、`Content-Language` 以及 `Content-Type` 这几个请求头在[简单 CO

==============================================================================
[advanced/settings.md]  行数=326  命中=28
     1| # 设置和环境变量 { #settings-and-environment-variables }
     7| 因此，通常会将它们提供为由应用程序读取的环境变量。
     9| **环境变量**（也称为 **env var**）是存在于 Python 代码之外、操作系统中的值，可以由你的应用和其他程序读取。
    11| 你可以在运行命令时为该命令创建环境变量。你将在下面看到特定于平台的命令。
    15| 阅读[环境变量指南](https://tiangolo.com/guides/environment-variables/)以详细了解环境变量的工作方式。
    21| 这些环境变量只能处理文本字符串，因为它们在 Python 之外，并且必须与其他程序及系统的其余部分兼容（甚至与不同的操作系统，如 Linux、Windows、macOS）。
    23| 这意味着，在 Python 中从环境变量读取的任何值都是 `str` 类型，任何到不同类型的转换或任何验证都必须在代码中完成。
    25| ## Pydantic 的 `Settings` { #pydantic-settings }
    27| 幸运的是，Pydantic 提供了一个很好的工具来处理来自环境变量的这些设置：[Pydantic：Settings 管理](https://pydantic.dev/docs/validation/latest/concepts/pydantic_settings/)。
    53| ### 创建 `Settings` 对象 { #create-the-settings-object }

==============================================================================
[tutorial/background-tasks.md]  行数=84  命中=18
     1| # 后台任务 { #background-tasks }
     3| 你可以定义在返回响应后运行的后台任务。
    14| ## 使用 `BackgroundTasks` { #using-backgroundtasks }
    16| 首先导入 `BackgroundTasks` 并在 *路径操作函数* 中使用类型声明 `BackgroundTasks` 定义一个参数：
    20| **FastAPI** 会创建一个 `BackgroundTasks` 类型的对象并作为该参数传入。
    24| 创建要作为后台任务运行的函数。
    36| ## 添加后台任务 { #add-the-background-task }
    38| 在你的 *路径操作函数* 里，用 `.add_task()` 方法将任务函数传到 *后台任务* 对象中：
    50| 使用 `BackgroundTasks` 也适用于依赖注入系统，你可以在多个级别声明 `BackgroundTasks` 类型的参数：在 *路径操作函数* 里，在依赖中(可依赖)，在子依赖中，等等。
    52| **FastAPI** 知道在每种情况下该做什么以及如何复用同一对象，因此所有后台任务被合并在一起并且随后在后台运行：

==============================================================================
[tutorial/testing.md]  行数=193  命中=25
     1| # 测试 { #testing }
     3| 感谢 [Starlette](https://starlette.dev/testclient/)，测试**FastAPI** 应用轻松又愉快。
     9| ## 使用 `TestClient` { #using-testclient }
    13| 要使用 `TestClient`，先要安装 [`httpx`](https://www.python-httpx.org)。
    23| 导入 `TestClient`。
    25| 通过传入你的**FastAPI**应用创建一个 `TestClient` 。
    29| 像使用 `httpx` 那样使用 `TestClient` 对象。
    37| 注意测试函数是普通的 `def`，不是 `async def`。
    47| 你也可以用 `from starlette.testclient import TestClient`。
    55| 除了发送请求之外，如果你还想测试时在FastAPI应用中调用 `async` 函数（例如异步数据库函数）， 可以在高级教程中看下[异步测试](../advanced/async-tests.md)。

==============================================================================
[advanced/custom-response.md]  行数=272  命中=38
     5| 你可以像在 [直接返回响应](response-directly.md) 中那样，直接返回 `Response` 来重载它。
     7| 但如果你直接返回一个 `Response`（或其任意子类，比如 `JSONResponse`），返回的数据不会自动转换（即使你声明了 `response_model`），也不会自动生成文档（例如，在生成的 OpenAPI 中，HTTP 头 `Content-Type` 里的特定「媒体类型」不会被包含
     9| 你还可以在 *路径操作装饰器* 中通过 `response_class` 参数声明要使用的 `Response`（例如任意 `Response` 子类）。
    11| 你从 *路径操作函数* 中返回的内容将被放在该 `Response` 中。
    25| 如果你没有声明响应模型，FastAPI 会使用 [JSON 兼容编码器](../tutorial/encoder.md) 中解释的 `jsonable_encoder`，并把结果放进一个 `JSONResponse`。
    27| 如果你在 `response_class` 中声明了一个 JSON 媒体类型（`application/json`）的类（比如 `JSONResponse`），你返回的数据会使用你在 *路径操作装饰器* 中声明的任意 Pydantic `response_model` 自动转换（和过滤）。但数据不会
    37| 使用 `HTMLResponse` 来从 **FastAPI** 中直接返回一个 HTML 响应。
    39| * 导入 `HTMLResponse`。
    40| * 将 `HTMLResponse` 作为你的 *路径操作* 的 `response_class` 参数传入。
    54| ### 返回一个 `Response` { #return-a-response }

==============================================================================
[advanced/websockets.md]  行数=186  命中=36
     1| # WebSockets { #websockets }
     3| 您可以在 **FastAPI** 中使用 [WebSockets](https://developer.mozilla.org/en-US/docs/Web/API/WebSockets_API)。
     5| ## 安装 `websockets` { #install-websockets }
     7| 将 `websockets`（一个让使用“WebSocket”协议更容易的 Python 库）添加到你的项目中：
    12| $ uv add websockets
    19| ## WebSockets 客户端 { #websockets-client }
    25| 要使用 WebSockets 与后端进行通信，您可能会使用前端的工具。
    27| 或者，您可能有一个原生移动应用程序，直接使用原生代码与 WebSocket 后端通信。
    29| 或者，您可能有其他与 WebSocket 终端通信的方式。
    39| 但这是一种专注于 WebSockets 的服务器端并提供一个工作示例的最简单方式：

==============================================================================
[tutorial/sql-databases.md]  行数=357  命中=28
     1| # SQL（关系型）数据库 { #sql-relational-databases }
     3| **FastAPI** 并不要求你使用 SQL（关系型）数据库。你可以使用你想用的**任何数据库**。
     7| **SQLModel** 基于 [SQLAlchemy](https://www.sqlalchemy.org/) 和 Pydantic 构建。它由 **FastAPI** 的同一作者制作，旨在完美匹配需要使用**SQL 数据库**的 FastAPI 应用程序。
    11| 你可以使用任意其他你想要的 SQL 或 NoSQL 数据库类库（在某些情况下称为 <abbr title="Object Relational Mapper - 对象关系映射器: 一个花哨的术语，指一种库，其中某些类表示 SQL 表，而实例表示这些表中的行">"ORMs"</abbr>），FastA
    15| 由于 SQLModel 基于 SQLAlchemy，因此你可以轻松使用任何由 SQLAlchemy **支持的数据库**（这也让它们被 SQLModel 支持），例如：
    25| 之后，对于你的生产应用程序，你可能会想要使用像 **PostgreSQL** 这样的数据库服务器。
    33| 这是一个非常简单和简短的教程。如果你想了解一般的数据库、SQL 或更高级的功能，请查看 [SQLModel 文档](https://sqlmodel.tiangolo.com/)。
    56| 导入 `SQLModel` 并创建一个数据库模型：
    64| * `table=True` 会告诉 SQLModel 这是一个*表模型*，它应该表示 SQL 数据库中的一个**表**，而不仅仅是一个*数据模型*（就像其他常规的 Pydantic 类一样）。
    66| * `Field(primary_key=True)` 会告诉 SQLModel `id` 是 SQL 数据库中的**主键**（你可以在 SQLModel 文档中了解更多关于 SQL 主键的信息）。

==============================================================================
[tutorial/security/oauth2-jwt.md]  行数=277  命中=26
     1| # 使用密码（及哈希）的 OAuth2，基于 JWT 的 Bearer 令牌 { #oauth2-with-password-and-hashing-bearer-with-jwt-tokens }
     3| 现在我们已经有了完整的安全流程，接下来用 <abbr title="JSON Web Tokens - JSON Web 令牌">JWT</abbr> 令牌和安全的密码哈希，让应用真正安全起来。
     9| ## 关于 JWT { #about-jwt }
    11| JWT 意为 “JSON Web Tokens”。
    27| 如果你想动手体验 JWT 令牌并了解它的工作方式，请访问 [https://jwt.io](https://jwt.io/)。
    29| ## 安装 `PyJWT` { #install-pyjwt }
    31| 我们需要安装 `PyJWT`，以便在 Python 中生成和校验 JWT 令牌。
    49| 可以在 [PyJWT 安装文档](https://pyjwt.readthedocs.io/en/latest/installation.html)中了解更多。
   131| ## 处理 JWT 令牌 { #handle-jwt-tokens }
   135| 创建一个用于对 JWT 令牌进行签名的随机密钥。

==============================================================================
[deployment/docker.md]  行数=615  命中=79
     1| # 容器中的 FastAPI - Docker { #fastapi-in-containers-docker }
     4| 部署 FastAPI 应用时，常见做法是构建一个**Linux 容器镜像**。通常使用 [**Docker**](https://www.docker.com/) 实现。然后你可以用几种方式之一部署该镜像。
    10| 赶时间并且已经了解这些？直接跳到下面的 [`Dockerfile` 👇](#build-a-docker-image-for-fastapi)。
    15| <summary>Dockerfile 预览 👀</summary>
    17| ```Dockerfile
    46| ## 什么是容器镜像 { #what-is-a-container-image }
    48| **容器**是从**容器镜像**运行的。
    50| 容器镜像是容器中所有文件、环境变量以及应该运行的默认命令/程序的一个**静态**版本。这里的**静态**指容器**镜像**本身并不在运行，仅仅是被打包的文件和元数据。
    52| 与存放静态内容的“**容器镜像**”相对，“**容器**”通常指一个正在运行的实例，即正在被**执行**的东西。
    54| 当**容器**启动并运行（从**容器镜像**启动）后，它可以创建或修改文件、环境变量等。这些更改只存在于该容器中，不会持久化到底层的容器镜像中（不会写回磁盘）。

==============================================================================
[tutorial/response-status-code.md]  行数=101  命中=25
     1| # 响应状态码 { #response-status-code }
     3| 与指定响应模型的方式相同，在以下任意*路径操作*中，可以使用 `status_code` 参数声明用于响应的 HTTP 状态码：
    11| {* ../../docs_src/response_status_code/tutorial001_py310.py hl[6] *}
    15| 注意，`status_code` 是（`get`、`post` 等）**装饰器**方法中的参数。与之前的参数和请求体不同，不是*路径操作函数*的参数。
    19| `status_code` 参数接收表示 HTTP 状态码的数字。
    23| `status_code` 还能接收 `IntEnum` 类型，比如 Python 的 [`http.HTTPStatus`](https://docs.python.org/3/library/http.html#http.HTTPStatus)。
    29| * 在响应中返回状态码
    30| * 在 OpenAPI schema（以及用户界面）中将其记录为该状态码：
    36| 某些响应状态码表示响应没有响应体（参阅下一节）。
    42| ## 关于 HTTP 状态码 { #about-http-status-codes }

==============================================================================
[tutorial/request-files.md]  行数=176  命中=51
     1| # 请求文件 { #request-files }
     3| 你可以使用 `File` 定义由客户端上传的文件。
     7| 要接收上传的文件，请先安装 [`python-multipart`](https://github.com/Kludex/python-multipart)。
    15| 这是因为上传文件是以「表单数据」发送的。
    19| ## 导入 `File` { #import-file }
    21| 从 `fastapi` 导入 `File` 和 `UploadFile`：
    25| ## 定义 `File` 参数 { #define-file-parameters }
    27| 像为 `Body` 或 `Form` 一样创建文件参数：
    33| `File` 是直接继承自 `Form` 的类。
    35| 但要注意，从 `fastapi` 导入的 `Query`、`Path`、`File` 等项，实际上是返回特定类的函数。

==============================================================================
[tutorial/security/first-steps.md]  行数=203  命中=23
     1| # 安全 - 第一步 { #security-first-steps }
     9| 我们可以用 **OAuth2** 在 **FastAPI** 中实现它。
    13| 我们直接使用 **FastAPI** 提供的安全工具。
    39| 这是因为 **OAuth2** 使用“表单数据”来发送 `username` 和 `password`。
    93| `password` “流”（flow）是 OAuth2 定义的处理安全与身份验证的一种方式。
    95| OAuth2 的设计目标是让后端或 API 与负责用户认证的服务器解耦。
   102| * 前端（运行在用户浏览器中）把 `username` 和 `password` 发送到我们 API 中的特定 URL（使用 `tokenUrl="token"` 声明）。
   115| ## **FastAPI** 的 `OAuth2PasswordBearer` { #fastapis-oauth2passwordbearer }
   117| **FastAPI** 在不同抽象层级提供了多种安全工具。
   119| 本示例将使用 **OAuth2** 的 **Password** 流程并配合 **Bearer** 令牌，通过 `OAuth2PasswordBearer` 类来实现。
