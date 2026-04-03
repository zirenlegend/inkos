import type { GenreProfile } from "../models/genre-profile.js";
import type { LengthSpec } from "../models/length-governance.js";
import type { AuditIssue } from "./continuity.js";
import type { ContextPackage, RuleStack } from "../models/input-governance.js";
import type { LLMMessage, LLMResponse } from "../llm/provider.js";
import { readGenreProfile, readBookLanguage, readBookRules } from "./rules-reader.js";
import { countChapterLength } from "../utils/length-metrics.js";
import { buildGovernedMemoryEvidenceBlocks } from "../utils/governed-context.js";
import { filterSummaries } from "../utils/context-filter.js";
import {
  buildGovernedCharacterMatrixWorkingSet,
  buildGovernedHookWorkingSet,
  mergeTableMarkdownByKey,
} from "../utils/governed-working-set.js";
import { applyLocalFixPatches, parseLocalFixPatches } from "../utils/local-fix-patches.js";
import { readFile } from "node:fs/promises";
import { join } from "node:path";

export type ReviseMode = "polish" | "rewrite" | "rework" | "anti-detect" | "local-fix";

export const DEFAULT_REVISE_MODE: ReviseMode = "local-fix";

export interface ReviseOutput {
  readonly revisedContent: string;
  readonly wordCount: number;
  readonly fixedIssues: ReadonlyArray<string>;
  readonly updatedState: string;
  readonly updatedLedger: string;
  readonly updatedHooks: string;
  readonly tokenUsage?: {
    readonly promptTokens: number;
    readonly completionTokens: number;
    readonly totalTokens: number;
  };
}

export interface WriterRepairInput {
  readonly bookDir: string;
  readonly chapterContent: string;
  readonly chapterNumber: number;
  readonly issues: ReadonlyArray<AuditIssue>;
  readonly mode: ReviseMode;
  readonly genre?: string;
  readonly chapterIntent?: string;
  readonly contextPackage?: ContextPackage;
  readonly ruleStack?: RuleStack;
  readonly lengthSpec?: LengthSpec;
}

export interface WriterRepairRuntime {
  readonly projectRoot: string;
  readonly chat: (
    messages: ReadonlyArray<LLMMessage>,
    options?: { readonly temperature?: number; readonly maxTokens?: number },
  ) => Promise<LLMResponse>;
}

const MODE_DESCRIPTIONS: Record<ReviseMode, { readonly zh: string; readonly en: string }> = {
  polish: {
    zh: "润色：只改表达、节奏、段落呼吸，不改事实与剧情结论。禁止：增删段落、改变人名/地名/物品名、增加新情节或新对话、改变因果关系。只允许：替换用词、调整句序、修改标点节奏",
    en: "Polish: Improve wording, rhythm, and paragraph flow only — do not alter facts or plot outcomes. Forbidden: adding/removing paragraphs, changing names/places/items, introducing new scenes or dialogue, altering cause-and-effect. Allowed: word choice, sentence reordering, punctuation adjustments",
  },
  rewrite: {
    zh: "改写：允许重组问题段落、调整画面和叙述力度，但优先保留原文的绝大部分句段。除非问题跨越整章，否则禁止整章推倒重写；只能围绕问题段落及其直接上下文改写，同时保留核心事实与人物动机",
    en: "Rewrite: Restructure problematic paragraphs and strengthen imagery, but keep most of the original text intact. Unless issues span the entire chapter, do not rewrite from scratch — only rework the flagged passages and their immediate surroundings while preserving core facts and character motivations",
  },
  rework: {
    zh: "重写：可重构场景推进和冲突组织，但不改主设定和大事件结果",
    en: "Rework: You may restructure scene progression and conflict organization, but do not change core settings or major event outcomes",
  },
  "anti-detect": {
    zh: `反检测改写：在保持剧情不变的前提下，降低AI生成可检测性。

改写手法（附正例）：
1. 打破句式规律：连续短句 → 长短交替，句式不可预测
2. 口语化替代：✗"然而事情并没有那么简单" → ✓"哪有那么便宜的事"
3. 减少"了"字密度：✗"他走了过去，拿了杯子" → ✓"他走过去，端起杯子"
4. 转折词降频：✗"虽然…但是…" → ✓ 用角色内心吐槽或直接动作切换
5. 情绪外化：✗"他感到愤怒" → ✓"他捏碎了茶杯，滚烫的茶水流过指缝"
6. 删掉叙述者结论：✗"这一刻他终于明白了力量" → ✓ 只写行动，让读者自己感受
7. 群像反应具体化：✗"全场震惊" → ✓"老陈的烟掉在裤子上，烫得他跳起来"
8. 段落长度差异化：不再等长段落，有的段只有一句话，有的段七八行
9. 消灭"不禁""仿佛""宛如"等AI标记词：换成具体感官描写`,
    en: `Anti-detection rewrite: Keep the plot unchanged while reducing detectable AI patterns.

Techniques (with examples):
1. Break sentence-length patterns: avoid uniform short or long sentences — vary unpredictably
2. Use natural idioms: ✗"However, the situation was far from simple" → ✓"No such luck"
3. Cut filler words: ✗"He walked over and grabbed the cup" → ✓"He crossed the room, snatched the cup"
4. Reduce mechanical transitions: ✗"Although…however…" → ✓ Use character reactions or hard scene cuts instead
5. Externalize emotion: ✗"He felt angry" → ✓"His fist hit the table and the teacup jumped"
6. Delete narrator conclusions: ✗"In that moment he finally understood power" → ✓ Show the action, let the reader feel it
7. Specific crowd reactions: ✗"Everyone was shocked" → ✓"Old Chen's cigarette fell on his trousers and he yelped"
8. Vary paragraph length: some paragraphs one line, others seven or eight
9. Eliminate AI-marker phrases ("couldn't help but", "as if", "it was as though"): replace with concrete sensory detail`,
  },
  "local-fix": {
    zh: "局部修复：只修改审稿意见指出的具体句子或段落，其余所有内容必须原封不动保留。修改范围限定在问题句子及其前后各一句。禁止改动无关段落",
    en: "Local fix: Only modify the specific sentences or paragraphs flagged by the review — everything else must remain untouched. Scope changes to the problem sentence plus at most one sentence before and after. Do not alter unrelated paragraphs",
  },
};

export async function repairChapterWithWriter(
  runtime: WriterRepairRuntime,
  input: WriterRepairInput,
): Promise<ReviseOutput> {
  const {
    bookDir,
    chapterContent,
    chapterNumber,
    issues,
    mode = DEFAULT_REVISE_MODE,
    genre,
    chapterIntent,
    contextPackage,
    ruleStack,
    lengthSpec,
  } = input;

  const [currentState, ledger, hooks, styleGuideRaw, volumeOutline, storyBible, characterMatrix, chapterSummaries, parentCanon, fanficCanon] = await Promise.all([
    readFileSafe(join(bookDir, "story/current_state.md")),
    readFileSafe(join(bookDir, "story/particle_ledger.md")),
    readFileSafe(join(bookDir, "story/pending_hooks.md")),
    readFileSafe(join(bookDir, "story/style_guide.md")),
    readFileSafe(join(bookDir, "story/volume_outline.md")),
    readFileSafe(join(bookDir, "story/story_bible.md")),
    readFileSafe(join(bookDir, "story/character_matrix.md")),
    readFileSafe(join(bookDir, "story/chapter_summaries.md")),
    readFileSafe(join(bookDir, "story/parent_canon.md")),
    readFileSafe(join(bookDir, "story/fanfic_canon.md")),
  ]);

  const genreId = genre ?? "other";
  const [{ profile: gp }, bookLanguage] = await Promise.all([
    readGenreProfile(runtime.projectRoot, genreId),
    readBookLanguage(bookDir),
  ]);
  const parsedRules = await readBookRules(bookDir);
  const bookRules = parsedRules?.rules ?? null;

  const isEnglish = (bookLanguage ?? gp.language) === "en";
  const resolvedLanguage = isEnglish ? "en" : "zh";

  const styleGuide = styleGuideRaw !== "(文件不存在)"
    ? styleGuideRaw
    : (parsedRules?.body ?? (resolvedLanguage === "en" ? "(no style guide)" : "(无文风指南)"));

  const suggestionLabel = resolvedLanguage === "en" ? "Suggestion" : "建议";
  const issueList = issues
    .map((issue) => `- [${issue.severity}] ${issue.category}: ${issue.description}\n  ${suggestionLabel}: ${issue.suggestion}`)
    .join("\n");

  const numericalRule = gp.numericalSystem
    ? resolvedLanguage === "en"
      ? "\n3. Numerical errors must be fixed precisely — cross-check before and after"
      : "\n3. 数值错误必须精确修正，前后对账"
    : "";
  const protagonistBlock = bookRules?.protagonist
    ? resolvedLanguage === "en"
      ? `\n\nProtagonist lock: ${bookRules.protagonist.name} — ${bookRules.protagonist.personalityLock.join(", ")}. Revisions must not violate the protagonist profile.`
      : `\n\n主角人设锁定：${bookRules.protagonist.name}，${bookRules.protagonist.personalityLock.join("、")}。修改不得违反人设。`
    : "";
  const lengthGuardrail = lengthSpec
    ? resolvedLanguage === "en"
      ? "\n8. Keep the chapter word count within the target range; only allow minor deviation when fixing critical issues truly requires it"
      : "\n8. 保持章节字数在目标区间内；只有在修复关键问题确实需要时才允许轻微偏离"
    : "";
  const governedMode = Boolean(chapterIntent && contextPackage && ruleStack);
  const hooksWorkingSet = governedMode && contextPackage
    ? buildGovernedHookWorkingSet({
        hooksMarkdown: hooks,
        contextPackage,
        chapterNumber,
        language: resolvedLanguage,
      })
    : hooks;
  const chapterSummariesWorkingSet = governedMode
    ? filterSummaries(chapterSummaries, chapterNumber)
    : chapterSummaries;
  const characterMatrixWorkingSet = governedMode
    ? buildGovernedCharacterMatrixWorkingSet({
        matrixMarkdown: characterMatrix,
        chapterIntent: chapterIntent ?? volumeOutline,
        contextPackage: contextPackage!,
        protagonistName: bookRules?.protagonist?.name,
      })
    : characterMatrix;

  const outputFormat = mode === "local-fix"
    ? resolvedLanguage === "en"
      ? `=== FIXED_ISSUES ===
(List each fix on its own line; if a safe local fix is not possible, explain here)

=== PATCHES ===
(Output only the local patches that need replacing — do not rewrite the whole chapter. Format below; repeat the PATCH block as needed)
--- PATCH 1 ---
TARGET_TEXT:
(Exact quote from the original that uniquely matches the target passage)
REPLACEMENT_TEXT:
(Replacement text for this passage)
--- END PATCH ---

=== UPDATED_STATE ===
(Full updated state card)
${gp.numericalSystem ? "\n=== UPDATED_LEDGER ===\n(Full updated resource ledger)" : ""}
=== UPDATED_HOOKS ===
(Full updated hooks board)`
      : `=== FIXED_ISSUES ===
(逐条说明修正了什么，一行一条；如果无法安全定点修复，也在这里说明)

=== PATCHES ===
(只输出需要替换的局部补丁，不得输出整章重写。格式如下，可重复多个 PATCH 区块)
--- PATCH 1 ---
TARGET_TEXT:
(必须从原文中精确复制、且能唯一命中的原句或原段)
REPLACEMENT_TEXT:
(替换后的局部文本)
--- END PATCH ---

=== UPDATED_STATE ===
(更新后的完整状态卡)
${gp.numericalSystem ? "\n=== UPDATED_LEDGER ===\n(更新后的完整资源账本)" : ""}
=== UPDATED_HOOKS ===
(更新后的完整伏笔池)`
    : resolvedLanguage === "en"
      ? `=== FIXED_ISSUES ===
(List each fix on its own line)

=== REVISED_CONTENT ===
(Full revised chapter content)

=== UPDATED_STATE ===
(Full updated state card)
${gp.numericalSystem ? "\n=== UPDATED_LEDGER ===\n(Full updated resource ledger)" : ""}
=== UPDATED_HOOKS ===
(Full updated hooks board)`
      : `=== FIXED_ISSUES ===
(逐条说明修正了什么，一行一条)

=== REVISED_CONTENT ===
(修正后的完整正文)

=== UPDATED_STATE ===
(更新后的完整状态卡)
${gp.numericalSystem ? "\n=== UPDATED_LEDGER ===\n(更新后的完整资源账本)" : ""}
=== UPDATED_HOOKS ===
(更新后的完整伏笔池)`;

  const localFixRules = resolvedLanguage === "en"
    ? "\n9. local-fix must only output local patches — full-chapter rewrites are forbidden; TARGET_TEXT must uniquely match a passage in the original\n10. If large-scale rewriting is needed, state that a safe local-fix is not possible and leave PATCHES empty"
    : "\n9. local-fix 只能输出局部补丁，禁止输出整章改写；TARGET_TEXT 必须能在原文中唯一命中\n10. 如果需要大面积改写，说明无法安全 local-fix，并让 PATCHES 留空";

  const systemPrompt = resolvedLanguage === "en"
    ? `You are a professional ${gp.name} web-fiction revision editor. Your task is to revise the chapter according to the review notes.${protagonistBlock}

Revision mode: ${MODE_DESCRIPTIONS[mode].en}

Revision principles:
1. Control the scope of changes according to the mode
2. Fix root causes — do not apply superficial polish${numericalRule}
4. Hook status must stay in sync with the hooks board
5. Do not alter the plot direction or core conflicts
6. Preserve the original language style and rhythm
7. After revising, update the state card${gp.numericalSystem ? ", ledger" : ""}, and hooks board
${lengthGuardrail}
${mode === "local-fix" ? localFixRules : ""}

ALL output — FIXED_ISSUES, ${mode === "local-fix" ? "PATCHES" : "REVISED_CONTENT"}, UPDATED_STATE, UPDATED_HOOKS — MUST be written entirely in English.

Output format:

${outputFormat}`
    : `你是一位专业的${gp.name}网络小说修稿编辑。你的任务是根据审稿意见对章节进行修正。${protagonistBlock}

修稿模式：${MODE_DESCRIPTIONS[mode].zh}

修稿原则：
1. 按模式控制修改幅度
2. 修根因，不做表面润色${numericalRule}
4. 伏笔状态必须与伏笔池同步
5. 不改变剧情走向和核心冲突
6. 保持原文的语言风格和节奏
7. 修改后同步更新状态卡${gp.numericalSystem ? "、账本" : ""}、伏笔池
${lengthGuardrail}
${mode === "local-fix" ? localFixRules : ""}

输出格式：

${outputFormat}`;

  const en = resolvedLanguage === "en";

  const ledgerBlock = gp.numericalSystem
    ? `\n## ${en ? "Resource Ledger" : "资源账本"}\n${ledger}`
    : "";
  const governedMemoryBlocks = contextPackage
    ? buildGovernedMemoryEvidenceBlocks(contextPackage, resolvedLanguage)
    : undefined;
  const hooksBlock = governedMemoryBlocks?.hooksBlock
    ?? `\n## ${en ? "Hooks Board" : "伏笔池"}\n${hooksWorkingSet}\n`;
  const outlineBlock = volumeOutline !== "(文件不存在)"
    ? `\n## ${en ? "Volume Outline" : "卷纲"}\n${volumeOutline}\n`
    : "";
  const bibleBlock = !governedMode && storyBible !== "(文件不存在)"
    ? `\n## ${en ? "World-building" : "世界观设定"}\n${storyBible}\n`
    : "";
  const matrixBlock = characterMatrixWorkingSet !== "(文件不存在)"
    ? `\n## ${en ? "Character Interaction Matrix" : "角色交互矩阵"}\n${characterMatrixWorkingSet}\n`
    : "";
  const summariesBlock = governedMemoryBlocks?.summariesBlock
    ?? (chapterSummariesWorkingSet !== "(文件不存在)"
      ? `\n## ${en ? "Chapter Summaries" : "章节摘要"}\n${chapterSummariesWorkingSet}\n`
      : "");
  const volumeSummariesBlock = governedMemoryBlocks?.volumeSummariesBlock ?? "";
  const canonBlock = parentCanon !== "(文件不存在)"
    ? en
      ? `\n## Parent Canon Reference (Revision Only)\nThis book is a spin-off. Revisions must respect canon constraints and must not alter canon facts.\n${parentCanon}\n`
      : `\n## 正传正典参照（修稿专用）\n本书为番外作品。修改时参照正典约束，不可改变正典事实。\n${parentCanon}\n`
    : "";
  const fanficCanonBlock = fanficCanon !== "(文件不存在)"
    ? en
      ? `\n## Fanfic Canon Reference (Revision Only)\nThis book is a fanfic. Revisions must respect canon character profiles and world rules. Character dialogue must preserve speech quirks from the source material.\n${fanficCanon}\n`
      : `\n## 同人正典参照（修稿专用）\n本书为同人作品。修改时参照正典角色档案和世界规则，不可违反正典事实。角色对话必须保留原作语癖。\n${fanficCanon}\n`
    : "";
  const reducedControlBlock = chapterIntent && contextPackage && ruleStack
    ? buildReducedControlBlock(chapterIntent, contextPackage, ruleStack, resolvedLanguage)
    : "";
  const lengthGuidanceBlock = lengthSpec
    ? en
      ? `\n## Word Count Guardrail\nTarget: ${lengthSpec.target}\nAllowed range: ${lengthSpec.softMin}-${lengthSpec.softMax}\nHard limits: ${lengthSpec.hardMin}-${lengthSpec.hardMax}\nIf the revised text exceeds the allowed range, prioritize trimming redundant explanations, repeated actions, and weak-information sentences — do not add subplots or remove core facts.\n`
      : `\n## 字数护栏\n目标字数：${lengthSpec.target}\n允许区间：${lengthSpec.softMin}-${lengthSpec.softMax}\n极限区间：${lengthSpec.hardMin}-${lengthSpec.hardMax}\n如果修正后超出允许区间，请优先压缩冗余解释、重复动作和弱信息句，不得新增支线或删掉核心事实。\n`
    : "";
  const styleGuideBlock = reducedControlBlock.length === 0
    ? `\n## ${en ? "Style Guide" : "文风指南"}\n${styleGuide}`
    : "";

  const userPrompt = en
    ? `Please revise Chapter ${chapterNumber}.

## Review Issues
${issueList}

## Current State Card
${currentState}
${ledgerBlock}
${hooksBlock}${volumeSummariesBlock}${reducedControlBlock || outlineBlock}${bibleBlock}${matrixBlock}${summariesBlock}${canonBlock}${fanficCanonBlock}${styleGuideBlock}${lengthGuidanceBlock}

## Chapter to Revise
${chapterContent}`
    : `请修正第${chapterNumber}章。

## 审稿问题
${issueList}

## 当前状态卡
${currentState}
${ledgerBlock}
${hooksBlock}${volumeSummariesBlock}${reducedControlBlock || outlineBlock}${bibleBlock}${matrixBlock}${summariesBlock}${canonBlock}${fanficCanonBlock}${styleGuideBlock}${lengthGuidanceBlock}

## 待修正章节
${chapterContent}`;

  const response = await runtime.chat(
    [
      { role: "system", content: systemPrompt },
      { role: "user", content: userPrompt },
    ],
    { temperature: 0.3, maxTokens: mode === "local-fix" ? 8192 : 16384 },
  );

  const output = parseRepairOutput(response.content, gp, mode, chapterContent, resolvedLanguage);
  const mergedOutput = governedMode
    ? {
        ...output,
        updatedHooks: mergeTableMarkdownByKey(hooks, output.updatedHooks, [0]),
      }
    : output;
  const wordCount = lengthSpec
    ? countChapterLength(mergedOutput.revisedContent, lengthSpec.countingMode)
    : mergedOutput.wordCount;

  return { ...mergedOutput, wordCount, tokenUsage: response.usage };
}

function parseRepairOutput(
  content: string,
  gp: GenreProfile,
  mode: ReviseMode,
  originalChapter: string,
  language: "zh" | "en",
): ReviseOutput {
  const extract = (tag: string): string => {
    const regex = new RegExp(
      `=== ${tag} ===\\s*([\\s\\S]*?)(?==== [A-Z_]+ ===|$)`,
    );
    const match = content.match(regex);
    return match?.[1]?.trim() ?? "";
  };

  const stateNotUpdated = language === "en" ? "(state card not updated)" : "(状态卡未更新)";
  const ledgerNotUpdated = language === "en" ? "(ledger not updated)" : "(账本未更新)";
  const hooksNotUpdated = language === "en" ? "(hooks board not updated)" : "(伏笔池未更新)";

  const fixedIssues = extract("FIXED_ISSUES")
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0);

  if (mode === "local-fix") {
    const patches = parseLocalFixPatches(extract("PATCHES"));
    const patchResult = applyLocalFixPatches(originalChapter, patches);

    return {
      revisedContent: patchResult.revisedContent,
      wordCount: patchResult.revisedContent.length,
      fixedIssues: patchResult.applied ? fixedIssues : [],
      updatedState: extract("UPDATED_STATE") || stateNotUpdated,
      updatedLedger: gp.numericalSystem
        ? (extract("UPDATED_LEDGER") || ledgerNotUpdated)
        : "",
      updatedHooks: extract("UPDATED_HOOKS") || hooksNotUpdated,
    };
  }

  const revisedContent = extract("REVISED_CONTENT");
  return {
    revisedContent,
    wordCount: revisedContent.length,
    fixedIssues,
    updatedState: extract("UPDATED_STATE") || stateNotUpdated,
    updatedLedger: gp.numericalSystem
      ? (extract("UPDATED_LEDGER") || ledgerNotUpdated)
      : "",
    updatedHooks: extract("UPDATED_HOOKS") || hooksNotUpdated,
  };
}

async function readFileSafe(path: string): Promise<string> {
  try {
    return await readFile(path, "utf-8");
  } catch {
    return "(文件不存在)";
  }
}

function buildReducedControlBlock(
  chapterIntent: string,
  contextPackage: ContextPackage,
  ruleStack: RuleStack,
  language: "zh" | "en",
): string {
  const selectedContext = contextPackage.selectedContext
    .map((entry) => `- ${entry.source}: ${entry.reason}${entry.excerpt ? ` | ${entry.excerpt}` : ""}`)
    .join("\n");
  const overrides = ruleStack.activeOverrides.length > 0
    ? ruleStack.activeOverrides
      .map((override) => `- ${override.from} -> ${override.to}: ${override.reason} (${override.target})`)
      .join("\n")
    : "- none";

  if (language === "en") {
    const none = "(none)";
    return `\n## Chapter Control Input (compiled by Planner/Composer)
${chapterIntent}

### Selected Context
${selectedContext || "- none"}

### Rule Stack
- Hard guardrails: ${ruleStack.sections.hard.join(", ") || none}
- Soft constraints: ${ruleStack.sections.soft.join(", ") || none}
- Diagnostic rules: ${ruleStack.sections.diagnostic.join(", ") || none}

### Active Overrides
${overrides}\n`;
  }

  return `\n## 本章控制输入（由 Planner/Composer 编译）
${chapterIntent}

### 已选上下文
${selectedContext || "- none"}

### 规则栈
- 硬护栏：${ruleStack.sections.hard.join("、") || "(无)"}
- 软约束：${ruleStack.sections.soft.join("、") || "(无)"}
- 诊断规则：${ruleStack.sections.diagnostic.join("、") || "(无)"}

### 当前覆盖
${overrides}\n`;
}
