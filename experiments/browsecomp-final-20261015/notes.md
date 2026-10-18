# BrowseComp-Plus 终轮准备记录

- 2026-10-15 决定：按用户拟租两块H800，安排同机2×80GB、TP=2共同承载一个Qwen3-32B BF16裁判；理由：保留官方参数并提供显存余量，不并行启动多个争抢两卡的判分进程。实机尚未连接或验证。
- 2026-10-15 决定：Mac负责sol生成/检索，OpenRouter负责监控，两块H800只负责官方判分；理由：本地GPU不能加速远端sol调用，职责分开才能并行工作。
- 2026-10-15 决定：先本地备包、再GPU双题冒烟，通过后Mac生成A-lean730且GPU顺序判已有A830/B230/A-lean100；理由：先确认官方判分能运行，再投入新一轮生成额度，同时利用已有答案让GPU有工作可做。
- 2026-10-15 决定：已有结果判完后若仍需等待生成，先取回evals并停止空闲GPU计算，最后补判A-lean新增730；理由：避免为等待远端模型付双卡空转费，缓存是否可保留以租赁平台实际规则为准。
- 2026-10-15 决定：明确bundle根目录布局及uv锁定环境，开始条件改为上一轮真实终态和最终补判/报告收尾；理由：修正审阅发现的环境与阶段衔接问题。本次只调整任务分配文档，未租机、未上传、未启动新一轮生成；其余选型历史和榜单措辞审阅意见仍保留在docs/issues.md。
- 2026-10-15 依据：vLLM 0.9.0.1单机多卡文档 https://docs.vllm.ai/en/v0.9.0.1/serving/distributed_serving.html ；NVIDIA H800规格分类 https://docs.nvidia.com/ai-enterprise/release-8/latest/infra-software/vgpu/reference/hopper-h800.html 。双卡配置是执行建议，不是实机性能或耗时验证。
# 最终官方复核执行记录

- 2026-10-15 决定：按用户最新夜间执行授权，在最终current_best冻结后由已登录Chrome自助启用AutoDL同机双H800；理由：用户已明确授权开机与既定预算，覆盖brief中旧的“不自行租机/没有GPU_SSH就等待”说明。现阶段未创建机器，不充值、不新增支付方式。
- 2026-10-15 决定：先实现search-only导出器，只有night-state=frozen才允许正式导出/打包；理由：官方判分不能影响本夜选型，原始轨迹和历史导出不改。
- 2026-10-15 离线验证：search工具返回的文档保留，只有read返回的文档排除，tool_call_counts仍保留search/read计数；未生成正式包、未启动官方评分。
- 2026-10-15 决定：官方源码与原0469490归档逐文件核对；理由：official目录不是独立Git仓库，在该目录运行git rev-parse会返回项目提交，不能将它误当官方版本证明。
- 2026-10-15 已核对当前官方结果CSV（85条）：AI21/最多8 agents为95.18%，Sail Research多agent为90.72%，Mixedbread与GPT-5.5 Question's Gambit均90.48%，GPT-5 BM25为57.59%。原始公开CSV在data，字段摘要及来源在public-reference-check.json。不同模型、检索、scaffold与预算不能据点估计自动认定排名或配对显著性。
- 2026-10-15 已修正上一轮报告中“现有最好”及“降本尚未验证”的过时措辞，旧分数和选择规则不变；明确A-lean已完成dev对照但未达原40%门槛、基线用量缺失不支持精确节省率。
- 2026-10-15 夜间已完成并提交晨报（eb0ba8e起，核账更正另提交），最终v0锁定，不再改选。export_v2实际导出A830/B230，search-only召回分别86.225856%/89.713768%，与旧导出相同。官方归档逐文件校验通过，私有gpu-bundle已生成；无需新生成730题，后置sol预算使用0/7000。下一步按授权从Chrome AutoDL准备双H800，先2题冒烟。
- 2026-10-15 AutoDL首次创建未成功，控制台实例0/余额未变；刷新后原H800主机可扩容从80GB变0GB。核查其他通用地区无H800。随后按同一双H800/19.95元每时创建成功，改用免费50GB数据盘＋120GB /dev/shm临时权重缓存（实例配额内存240GB）；付费数据盘为0。未改变模型、BF16、TP=2或官方参数。
- 2026-10-15 真实连接验证：两卡均NVIDIA H800 PCIe/81559MiB，driver595.71.05；GPU间为SYS跨NUMA，没有NVLink。/dev/shm120GB足以存储权重；锁定环境和私有判分包放/root/autodl-tmp/minimal-rag-final-20261015。平台自动关机已确认设为2026-10-15 14:50:00（页面服务器时间与本地LA一致），预计余额内消费，不充值。
- 2026-10-15 后置计划data/benchmarks/browsecomp-plus/final-20261015/final-plan.json固定11:45:14–15:45:14；计算至14:50自动关机，预留取回/清理时间。selected=v0/A，A830+B230已上传私有GPU；本阶段新sol0/7000。HF官方模型revision=9216db5781bf21249d130ec9da846c4624c16137。
- 2026-10-15 安装已后台启动：远端/root/autodl-tmp/gpu-setup.py，日志/root/autodl-tmp/gpu-setup.log，状态/root/autodl-tmp/minimal-rag-final-20261015/setup-state.json。本地SSH/SCP助手在data/benchmarks/browsecomp-plus/final-20261015/connection/remote.py（run/put/get），凭据仅在同目录600权限文件，不打印、不进Git。后续先检查锁环境/下载状态，ready后官方两题冒烟，保留有效eval，再A830/B230完整判分；结束取回evals、删除远端私有包并停止实例。不得因GPU分数改选或重新开启候选。
- 2026-10-15 安装启动纠正：一次SSH启动连接关闭；随后nohup返回2099但非登录SSH的PATH无python，日志确认未执行。通过只读检查找到/root/miniconda3/bin/python，使用该明确路径并仅为安装进程补充其bin路径后重启；PID2235已核实实际运行，setup-state进入locked_environment，uv0.13.1安装成功。没有复跑模型或扩展预算，之前日志和失败记录保留。
- 2026-10-15 安装网络恢复：官方Astral CDN在GPU仅约KB/s，Mac经同一官方GitHub来源8秒取得Python3.10.22原包，已转存并验证版本。uv.lock不改，改用该Python路径；Tevatron锁定commit67eb67604c0e7d1753fefe3b232ddfb84795f5ea由Mac取回，通过仅安装进程的Git URL映射传输本地原仓库，未改代码。转存tar的UID导致Git所有权检查失败，修正本任务仓库所有者为root后恢复，没有设置全局safe.directory例外。
- 2026-10-15 当前真实控制器/root/autodl-tmp/gpu-continue.py PID3123，setup-state确认locked_environment(PID3124)和official_weights(PID3125)都在运行；旧2235/2246和第一次并行2853/2855已停。uv已开始torch/CUDA等锁定包下载。权重改为Qwen官方ModelScope，17分片共65,524,328,560 bytes的元数据SHA256与HF固定revision全部相等；最终每文件还会校验实际SHA256，配置/分词器来自HF原固定revision。权重下载并发8，不涉及生成5/裁判1的实验并发设置。
- 2026-10-15 SSH凭据仍仅在受限本地文件；所有网络转存均为公共依赖或已授权的私有GPU包，没有上传公开存储。环境和下载尚未完成，官方两题/全量判分均尚未运行；14:50自动关机与15:45后置硬截止不延长。
- 2026-10-15 PyPI原下载链路同样慢，uv-cache数分钟仅16MB；停止旧环境安装3124及旧控制器3123，保留权重下载3125运行。控制器改为/root/autodl-tmp/gpu-env-recover.py PID3742：从官方uv.lock原样导出锁定requirements，仅暂分离Tevatron固定Git项；通过HTTPS Aliyun PyPI镜像逐项require-hashes安装，最后仍执行uv sync --frozen并验证版本。官方源码和uv.lock均不改，镜像不改变依赖版本。已观察到包下载显著推进且deepspeed0.17.2构建成功，尚未完成环境验证。
- 2026-10-15 官方运行器已上传但未启动：/root/autodl-tmp/official-run.py与gpu-control.json。setup-state=ready且确认模型/环境后才启动；固定冒烟ID131/1257，之后A830/B230，保留有效缓存与首次判决，仅补缺失/解析错误、每题最多3次模型响应。推理硬停14:45，平台关机14:50，预留取回清理；原始数据留私有GPU/data。状态在ROOT/official-state.json（含完整summary，巡检只输出状态/计数，勿输出原题/答案/裁判正文）。
- 2026-10-15 自动承接已建立：start-official-when-ready.py PID4781等待setup-state=ready，核对Python3.10、torch2.7.0、vLLM0.9.0.1、Transformers4.53.2、BF16、17权重分片及原脚本/uv.lock后，自动exec official-run.py。后者的单实例锁继承到评分子进程，防止续办重复启动。不要另开官方评测进程；只读ROOT/official-state.json及日志进度，完成后取回并清理。
- 2026-10-15 AutoDL官方学术加速仅在临时下载测试子进程中启用；32MiB单流约2.53MB/s，4流256MiB测试55秒未完成，没有证明比当前ModelScope下载更快，因此保留当前来源和已下载内容。没有修改全局网络、任何权限或模型参数。
- 2026-10-15 保护定时调整（未改变原授权预算/后置硬截止）：实际下载带宽使判分余量变小，依据用户已授权“现有AutoDL余额可用于本任务”，平台已确认关机由14:50调整为15:10，推理停止改为15:05，仍在原15:45:14截止和原余额69.90元内；没有充值或新增费用项目。即使从11:45准备起都计费，至15:10也≤68.17元，实际创建更晚；最终以平台账单为准。先前3小时/59.85元是助手额外设置的内部保护，不是另获追加预算。修改前UI显示13:31至15:10预计32.92元，已记录原保护值及修订理由。模型、温度、分母和新sol上限不变。
- 2026-10-15 当前等待器改为5598并重新读取新gpu-control（旧4781当时仍仅等待、未发起评分，已停）。环境控制器3742仍在锁定包安装，权重3125仍在下载；其磁盘脚本已在本地更新新保护时间，若在旧内层等待上限触发前需要恢复须使用最新脚本，不重跑任何有效判决。官方判分仍尚未开始。
- 2026-10-15 13:38 LA核对：锁定requirements的镜像安装已完成（记录vLLM0.9.0.1等版本），正在最后uv sync --frozen构建原Tevatron固定commit；权重缓存约42GiB，尚未完成实际分片校验。pypdfium2撤回提示仅记issues，本轮不更换官方锁定版本。
- 2026-10-15 13:44 LA实际环境验证通过：项目.venv中Python3.10.22、torch2.7.0+cu126、vLLM0.9.0.1、transformers4.53.2；CUDA12.6运行库，2个GPU，BF16支持=true。最后uv sync --frozen完成于13:40:03 LA；已安装原Tevatron commit。没有用基础镜像Python3.12/PyTorch2.8替代。权重实际校验仍待全部下载完成，官方评分尚未开始。
- 2026-10-15 正式结果完成：A772/830=93.01%，B194/230=84.35%，共1054个首次有效模型判决，另保留A4/B2个原生成失败；没有补判。官方流程从14:09:50至14:21:13 LA约11.4分钟；首个64题12.7秒。全部分片实际SHA256已通过，源码/uv.lock与官方原版一致。
- 2026-10-15 完整结果与运行证据已取回并验证SHA256：evals.tar.gz=b5e9c7a0a20e4ee5bcc79bbc12ccd2bd4ef00335e78fd2ad4195202699e0c0eb，runtime-evidence.tar.gz=dcf5f5b6d4c35bbb124a180a67d1cb136b664ec0e6c3bce846c6e66a98d8035f。一次并行SCP认证失败，另一包成功；同一现有凭据顺序重取成功，没有更改权限或凭据。1060个eval与原run的ID/答案/搜索文档逐项相等。
- 2026-10-15 收尾已完成：远端题目、答案、ground truth、全部eval/日志及含答案归档已删除并核验；仅保留原路径下的公开.venv、pyproject.toml、uv.lock和公共缓存。实例已确认“已关机”，付费盘0GB；保留免费环境，不必不可逆释放可复用依赖。平台定时关机已取消，本聊天续办已PAUSED，旧browsecomp仍不恢复。隐私裁剪的关机证明为data下gpu-stopped.jpg，不含SSH凭据。
- 2026-10-15 已生成并准备本地提交submission.json与5行英文scaffold，未发送/上传。官方摘要日期为服务器Asia/Shanghai的2026-10-16；提交元数据统一为客户端America/Los_Angeles的2026-10-15，原官方摘要和per_query_metrics完整保留。费用日结按10-15至10-16筛选暂无数据，按整个后置准备到关机核验窗口估算上界约58.5元，不能称已结算金额。
