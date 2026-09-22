from pathlib import Path
import json, hashlib
from xml.sax.saxutils import escape
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor, Color, white
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/pdf/FPE自洽算法_工程验证总结报告.pdf'
S=json.loads((ROOT/'runs/acceptance-v2/summary.json').read_text())
D=json.loads((ROOT/'reports/delivery.json').read_text())
pdfmetrics.registerFont(TTFont('CN','/System/Library/Fonts/Supplemental/Arial Unicode.ttf'))
W,H=595.276,841.89
M=43; CW=W-2*M
NAVY=HexColor('#17324D'); INK=HexColor('#24364A'); MUTED=HexColor('#5D6B79'); BLUE=HexColor('#176B89'); RED=HexColor('#A63C37'); LIGHT=HexColor('#EDF3F7'); LINE=HexColor('#D9E2EA')
c=canvas.Canvas(str(OUT),pagesize=(W,H),pageCompression=1)
c.setTitle('FPE 自洽算法：工程实现与验证总结')
c.setAuthor('随机分析证明项目')
c.setSubject('低维自适应Fokker-Planck求解器与真实图像生成可行性验证；完整科学验收未通过')
page=0;y=0

def para(text,size=10.2,color=INK,leading=None,space=9,x=M,width=CW):
 global y
 st=ParagraphStyle('p',fontName='CN',fontSize=size,leading=leading or size*1.58,textColor=color,wordWrap='CJK',splitLongWords=True)
 p=Paragraph(text,st); ww,hh=p.wrap(width,800)
 assert y-hh>48,(page,text[:70],y,hh)
 p.drawOn(c,x,y-hh); y-=hh+space

def title(text):
 global y
 para(text,15.5,NAVY,space=12)

def sub(text):
 para(text,11.4,BLUE,space=7)

def table(rows,widths,fs=9.1):
 global y
 st=ParagraphStyle('cell',fontName='CN',fontSize=fs,leading=fs*1.4,textColor=INK,wordWrap='CJK')
 hs=ParagraphStyle('head',parent=st,textColor=white)
 data=[[Paragraph(escape(str(v)),hs if i==0 else st) for v in row] for i,row in enumerate(rows)]
 t=Table(data,colWidths=widths,hAlign='LEFT')
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),('ROWBACKGROUNDS',(0,1),(-1,-1),[white,LIGHT]),('LINEBELOW',(0,-1),(-1,-1),.5,LINE)]))
 tw,th=t.wrap(CW,800); assert y-th>48,(page,'table',y,th)
 t.drawOn(c,M,y-th);y-=th+12

def image(path,x,width,height=None):
 global y
 im=ImageReader(str(path)); iw,ih=im.getSize(); height=height or width*ih/iw
 c.drawImage(im,x,y-height,width,height,mask='auto',preserveAspectRatio=True,anchor='c')
 return height

def start(kicker,heading):
 global page,y
 if page:c.showPage()
 page+=1;c.setFillColor(NAVY);c.rect(0,H-12,W,12,fill=1,stroke=0)
 c.setFont('CN',8.3);c.setFillColor(MUTED);c.drawString(M,H-38,'FPE 自洽算法  /  工程验证总结');c.drawRightString(W-M,H-38,'2026-09-22  ·  v0.1.0')
 c.setStrokeColor(LINE);c.line(M,43,W-M,43)
 c.setFont('CN',8);c.drawString(M,28,'工程研究记录 | 结果冻结于 2026-09-22');c.drawRightString(W-M,28,f'{page} / 7')
 y=H-68;para(kicker,9,BLUE,space=8);title(heading)

start('01 / 执行摘要','实现已落地，完整科学验收尚未通过')
para('本项目将 Fokker-Planck 方程的自洽思想实现为可训练概率流，并加入诊断、单因素调整、独立验证及回退机制。本轮同时检验低维方程求解与真实图片生成的工程可行性。',11)
para('当前结论：低维自适应机制有实测收益，但存在保留审查失败；图像分支尚未生成可辨认的自然物体。不能将当前版本称为成熟通用求解器或复杂图片生成器。',11,RED,space=16)
table([['检验对象','结果','可以支持的结论'],['低维自适应开发基准','15 / 15 通过','五类问题 × 三个种子，21 个时间点达到冻结阈值'],['未参与调参的最终审查','5 / 6 通过','一个二维耦合运行失败，完整工程验收未通过'],['空间扩阶受控对照','3 / 3 种子改善','共享预算上限下，最终 KL 中位数降低 96.24%'],['真实 CIFAR-10 图像','两种容量均失败','32×32 RGB 输出仍为噪声与模糊色块'],['实现与安装检查','116 项测试通过','本机 1 项平台专用跳过；云端对应测试通过，独立安装通过']],[122,105,CW-227])
sub('本轮交付了什么')
para('已交付 Python 安装包、五类核心模块、训练与演化命令、时间边缘分布的采样和密度查询接口、检查点与恢复记录、独立评价器、云端受控运行脚本，以及可重算的实验结果和失败记录。')
sub('如何理解“通过”与“未通过”')
para('45 个预定验收任务全部产生结果，进程执行没有缺失或异常终态；其中包含精度不达标和明确数值退出。单元测试通过验证实现机制，不能替代分布精度审查。所有种子和被拒候选均被保留，未通过放宽阈值获得成功结论。')
para('面向决策：可以作为继续研究的可复核工程基础；本轮不支持生产发布、严格误差认证、高维通用求解或复杂图像生成能力的主张。',10.4,RED)

start('02 / 理论与实现','把“自洽”变成可测量、可回退的算法')
sub('问题范围与理论来源')
para('主求解器限定为 1-2 维周期域 [0, 2π)<super>d</super>，扩散系数为 1，时间不变光滑势 V，严格正且初始密度与 score 可计算。每个 FPE 实例单独训练，不宣称跨实例通用模型。理论出发点是 Shen 等人的 Self-Consistency of the Fokker-Planck Equation（COLT 2022）[1]。')
para('直观上，速度场推动一团概率质量移动；它所产生的密度又反过来决定应有的速度。两者差异就是自洽残差。训练不断减小该差异，控制器只在独立内部验证支持改善时保留新版本。')
sub('同一次流积分传播四类量')
para('x′ = v；  ℓ′ = -div v<br/>s′ = -(Dv)<super>T</super>s - ∇div v<br/>q′ = ||v + ∇V + s||²',11.2,BLUE,leading=20,space=10)
para('其中 ℓ 为 log-density，s 为 score（对数密度的空间梯度），q 为累计残差。密度和 score 均从真实初值传播，每次更新重新计算完整轨迹，没有让一个自由 score 网络抵消误差。')
table([['模块','工程实现'],['问题与表示','Fourier 空间基 × Bernstein 时间基；扩阶精确保留旧速度'],['流与训练','Python 3.12 / JAX float64 / Optax；固定步长 RK4 与离散反向传播'],['演化控制','优先排查积分、采样与优化，再试空间或时间容量；一次只改一项'],['独立评价','NumPy/SciPy DOP853；解析解或保正守恒有限体积参考；细化检查'],['可追溯运行','doctor / train / evolve / evaluate / resume；参数、随机状态、父版本与理由留档']],[100,CW-100])
sub('理论保证没有被直接当作数值证书')
para('原文的 Wasserstein 结果使用二阶 Sobolev 自洽目标；本实现训练零阶残差，只在少量点诊断高阶量。对连续、光滑、正密度及精确流，另有条件熵界 KL(ρ<sub>t</sub> || ρ<sub>FPE,t</sub>) ≤ E[q(t)]/4。有限样本损失、离散积分和优化误差尚未获得严格认证；本轮没有证明自进化收敛。',9.6)

start('03 / 低维基准','开发基准通过，但不能替代独立审查')
para('每个案例使用种子 0、1、2，在预定 21 个时间点评价。下表为三个种子、所有时间点的最大误差；固定配置与自适应结果均完整保留。KL 衡量整体分布差异；TV 可理解为两种概率分布相差的质量规模。',10)
rows=[['案例','固定配置\n通过数','自适应\n通过数','自适应\n最大 KL','自适应\n最大 TV']]
labels={'uniform':'均匀初值 / 零势','heat1d':'一维热方程','heat2d':'二维可分离热方程','doublewell':'一维双势阱','coupled2d':'二维耦合势'}
for case,label in labels.items():
 items=[S['results'][f'adaptive-{case}-s{i}'] for i in range(3)]
 nf=sum(S['results'][f'fixed-{case}-s{i}']['passed'] for i in range(3))
 rows.append([label,f'{nf}/3','3/3',f"{max(v['maxima']['kl'] for v in items):.6g}",f"{max(v['maxima']['tv'] for v in items):.6g}"])
table(rows,[153,69,69,108,CW-399])
h=image(ROOT/'reports/acceptance.png',M,CW);y-=h+8
para('图 1｜全部开发基准种子。叉号为固定配置，圆点为自适应；虚线为冻结阈值。仅为对数显示将零值置于 10<super>-8</super>，上表保留真实值。图中不包含最终保留审查。',8.7,MUTED,space=13)
sub('冻结的精度与一致性要求')
para('KL ≤ 0.001；TV ≤ 0.02；质量误差 ≤ 0.0001；势阱质量误差 ≤ 0.01。参考网格细化 TV 变化 ≤ 0.0002；模型数值细化 TV 变化 ≤ 0.001。方向梯度与密度/score 交叉检查的归一化误差要求均为 0.0001。',9.6)
para('自适应基准的最大质量误差为 5.49×10<super>-14</super>，最大势阱质量误差约 0.002452。固定基线仅 9/15 通过，自适应为 15/15；这一改善仍不能覆盖下一页的最终审查失败。',9.6)

start('04 / 演化与最终审查','演化有效，但搜索稳定性尚不足')
para('默认 K=2、时间次数 p=3、batch=128、RK4 128 步、学习率 0.01；每候选最多 500 次更新。动作限于步数翻倍、batch 翻倍、学习率减半、空间阶数 +1、时间次数 +2。资格检查通过后，配对残差须至少降低 5%，且超过三倍标准误及数值裕量才晋升。连续三次拒绝即停机。',9.7)
sub('预声明欠拟合对照：三阶初值、初始 K=2')
rows=[['种子','自适应最终 KL','固定 K=2 最终 KL','实际时间：自适应 / 固定']]
for r in S['control']['seeds']:rows.append([r['seed'],f"{r['adaptive_final_kl']:.6g}",f"{r['fixed_final_kl']:.6g}",f"{r['adaptive_seconds']:.2f} / {r['fixed_seconds']:.2f} 秒"])
table(rows,[45,130,130,CW-305],9)
para('两组共用 120 秒及最多 12 个训练块的预算上限，包含编译、候选探索和内部验证；实际耗时不同，不能称为同实际算力。三个种子均改善，最终 KL 中位数降低 96.24%，超过预声明 30% 要求。参考评价在模型冻结后执行。',9.5)
sub('未参与调参的保留审查：5/6 通过')
rows=[['审查问题','种子','最大 KL','最大 TV','结论']]
for i in range(2):
 for seed in range(3):
  r=S['results'][f'audit-{i}-s{seed}'];rows.append(['热方程' if i==0 else '二维耦合势',seed,f"{r['maxima']['kl']:.6g}",f"{r['maxima']['tv']:.6g}",'通过' if r['passed'] else '失败'])
table(rows,[105,44,125,125,CW-399],8.8)
para('失败运行 audit-1-s1 的最差时刻为 t=1：KL=0.00314443、TV=0.03076995。21 个时刻中分别有 12 个、11 个超限；参考及模型细化检查通过。空间扩阶、时间升阶和积分细化候选连续被拒，保留初始模型并停止为平台期。不能仅凭此确定是容量还是优化问题。',9.5,RED)
para('压力测试也全部保留失败：高势垒达到 12 轮预算上限；高频初值停在平台期；刻意粗积分与小 batch 无合格初始模型而明确数值退出。若今后据审查结果改算法，该批须转入开发集，并建立新的未见审查批。',9.4)

start('05 / 真实图像试验','计算路径可运行，图片生成未通过')
para('独立图像分支在 3072 维欧氏空间求解 OU 输运，超出周期低维验收范围。数据来自官方 CIFAR-10 [2]：512 张训练图作为高斯混合中心，σ=0.2；256 张独立测试图用于事后对照。初始 score 已知，没有预训练生成模型或真实中间时刻 score 监督。',9.8)
sub('完整输出：独立高斯终端逆流采样，各 64 张，未挑图')
left=M; gap=15; width=(CW-gap)/2
c.setFont('CN',9.7);c.setFillColor(BLUE);c.drawString(left,y,'rank 32');c.drawString(left+width+gap,y,'rank 128');y-=10
h1=image(ROOT/'runs/image-seed0/independent_generated.png',left,width)
h2=image(ROOT/'runs/image-rank128-seed0/independent_generated.png',left+width+gap,width)
y-=max(h1,h2)+10
para('图 2｜原始实验输出，显示时裁剪到像素范围。两组均为彩色噪声和模糊色块，没有可辨认的自然物体；未经精选，未使用额外图像生成工具修饰。',8.8,MUTED)
table([['指标','rank 32','rank 128'],['更新数 / 种子','500 / 0','500 / 0'],['训练时间（秒）','106.47','239.44'],['独立残差 / 坐标','15.6813','10.4270'],['前向终端 KL 估计','3948.59','2177.91'],['128→256 步配对像素 RMSE','0.00001512','0.00002333'],['视觉结果','失败','失败']],[CW-218,109,109],9)
para('增加容量使每坐标残差降低 33.5%，却没有带来可辨认图像。这是同更新次数、不同实际计算成本的冷启动容量对照，不能称为保留旧函数的自动演化。细化变化很小，也不足以解释本轮视觉失败；该检查不等于高维误差证书。',9.4)
para('目标高斯混合本来即可直接采样，故本试验检验已知分布的输运能力，不能证明学会未知自然图像分布。没有 FID、新颖性证明或高分辨率测试；一个种子的两种架构失败也不构成理论不可能结论。',9.3,RED)

start('06 / 可靠性与成本','工程机制可复核，当前没有速度优势')
sub('实现与运行验证')
table([['检查','实测结果 / 边界'],['CPU 完整测试集','116 通过、1 个 Linux 专用跳过；该项已在云端通过'],['云端运行时测试','19 项通过；12 个作业均有终态，终态核对未见项目进程或 GPU 计算占用'],['方向梯度 / score 检查','归一化误差分别约 5.24×10^-13 / 1.32×10^-8'],['独立安装','新 Python 3.12 环境、非 editable wheel；16 模块来源和哈希、CLI/API 与恢复检查通过'],['数据与运行追溯','检查点、随机状态、父版本、拒绝理由、源码哈希及终态回执保存；所有45项任务有结果']],[127,CW-127],9.2)
para('云端使用独立环境与共享 GPU 锁；单任务限制和项目进程监控避免不受控重复提交。16 GiB 内存/显存限制为轮询监测上限，不是硬隔离配额。本地任务使用统一 7200 秒截止时间，包含终止和回收；无法确认回收会阻止继续提交。',9.3)
sub('同一组观测量精度下的实测成本（CPU，种子 0）')
rows=[['案例','训练+内部\n验证（秒）','密度评价\n（秒）','参考解\n（秒）','SDE含细化\n（秒）']]
for case in labels:
 r=json.loads((ROOT/f'runs/acceptance-v2/adaptive-{case}-s0/evaluation.json').read_text());cost=r['costs_seconds'];cl=r['classical_comparison']
 rows.append([labels[case],f"{cost['training_search_validation_total']:.2f}",f"{cost['density_queries_and_refinement']:.2f}",f"{cost['reference']:.4f}",f"{cl['sde_total_seconds']:.2f}"])
table(rows,[149,94,88,83,CW-414],8.7)
para('比较目标为同一组 Fourier 观测量和势阱质量的绝对误差 0.01，五例均达到该观测量精度；不以 SDE 的观测量误差冒充分布 KL/TV。热方程参考为解析解，势阱/耦合参考为有限体积法；成本包含各自的细化尝试。',9.3)
para('训练总时长已计入候选失败和编译；编译与首次执行合并记录，不虚构纯编译时间。模型独立采样 2048 点耗时约 0.005-0.071 秒，但快速重复采样不能抵消首次训练成本。本轮总成本高于经典对照，不支持效率优势。',9.3,RED)

start('07 / 结论与证据','下一步应优先辨别失败原因，再扩展范围')
sub('当前可交付的判断')
para('可以交付具备复核链条的研究实现，以及“有限动作的自适应在预声明低维欠拟合案例中有效”的实测结论。完整工程验收仍失败；低维误差证据、原论文理论保证、高维图像质量和自进化收敛证明必须分别讨论。')
sub('后续研究建议（本轮未执行）')
table([['优先级','待验证假设与建议实验','接受改进的前提'],['1','区分候选训练不足、学习率、采样噪声与三拒停止策略；开发集中做单因素受控实验','保持总预算和全部种子；不能使用审查 KL 回选模型'],['2','对失败耦合问题改善搜索稳定性后，生成新的幅度/相位审查组合','旧审查批转为开发数据；重新冻结阈值和新批次'],['3','图像分支研究适合空间结构的速度表示及早期时间变化；先从小图验证','完整输出与独立前向误差同时改善，再扩大分辨率'],['4','研究统计误差、离散误差及高阶残差的严格控制','在得到证明前，继续使用“工程证据”表述']],[48,259,CW-307],9)
sub('证据与复核入口')
para('实验主体：acceptance-v2，45 个任务，所有种子均纳入报告。早期 acceptance-v1 保留为开发记录；固定基线复用有源码及逐元素等价核验。最终审查后只修正任务截止时间的执行路径，数学函数、控制器、模型参数与冻结规格未变。',9.3)
para('随项目保存：acceptance.md（总体数据）、heldout-failure-review.md（独立失效分析）、image-feasibility.md / image-evidence-review.md（图片与核验）、summary.json（机器可读汇总）、delivery.json（独立安装）、full-tests-final.txt（测试记录）。本 PDF 嵌入关键数据与实际样本，可独立阅读；完整原始轨迹与检查点由项目归档保管。',9.1)
para('冻结规格 SHA-256：<br/><font size="7.5">584226cb3df139c509a1555f51ed5df7eeac390d4a1af9bbccb93de3a37ba815</font><br/>最终 wheel SHA-256：<br/><font size="7.5">'+D['artifact']['wheel_sha256']+'</font>',8.6,MUTED,leading=14)
sub('外部资料')
para('[1] Shen, Wang, Kale, Ribeiro, Karbasi &amp; Hassani. <i>Self-Consistency of the Fokker-Planck Equation</i>. COLT 2022, PMLR 178.<br/><link href="https://proceedings.mlr.press/v178/shen22a.html" color="#176B89">proceedings.mlr.press/v178/shen22a.html</link><br/>[2] CIFAR-10 官方数据与说明：<link href="https://www.cs.toronto.edu/~kriz/cifar.html" color="#176B89">cs.toronto.edu/~kriz/cifar.html</link>。下载档案经官方 MD5 校验；本报告结果来自项目实际运行，不来自原论文实验。',8.8)
c.save()
print(OUT)
print('pages',page,'bytes',OUT.stat().st_size,'sha256',hashlib.sha256(OUT.read_bytes()).hexdigest())
