import type { AuditIssue, AuditResult } from "../agents/continuity.js";
import type { ReviseOutput } from "../agents/reviser.js";
import type { WriteChapterOutput } from "../agents/writer.js";
import type { LengthSpec } from "../models/length-governance.js";

export interface ChapterReviewCycleUsage {
  readonly promptTokens: number;
  readonly completionTokens: number;
  readonly totalTokens: number;
}

export type ChapterRepairDecision = "none" | "local-fix" | "rewrite";

export interface ChapterAssessment {
  readonly auditResult: AuditResult;
  readonly repairIssues: ReadonlyArray<AuditIssue>;
  readonly repairDecision: ChapterRepairDecision;
  readonly aiTellCount: number;
  readonly blockingCount: number;
  readonly criticalCount: number;
}

export interface ChapterReviewCycleResult {
  readonly finalContent: string;
  readonly finalWordCount: number;
  readonly preAuditNormalizedWordCount: number;
  readonly revised: boolean;
  readonly auditResult: AuditResult;
  readonly totalUsage: ChapterReviewCycleUsage;
  readonly postReviseCount: number;
  readonly normalizeApplied: boolean;
}

export async function runChapterReviewCycle(params: {
  readonly initialOutput: Pick<WriteChapterOutput, "content" | "wordCount">;
  readonly initialRepairIssues?: ReadonlyArray<AuditIssue>;
  readonly lengthSpec: LengthSpec;
  readonly initialUsage: ChapterReviewCycleUsage;
  readonly assessChapter: (
    chapterContent: string,
    options?: {
      temperature?: number;
      initialRepairIssues?: ReadonlyArray<AuditIssue>;
    },
  ) => Promise<ChapterAssessment>;
  readonly repairChapter: (
    chapterContent: string,
    issues: ReadonlyArray<AuditIssue>,
    mode: Exclude<ChapterRepairDecision, "none">,
  ) => Promise<ReviseOutput>;
  readonly normalizeDraftLengthIfNeeded: (chapterContent: string) => Promise<{
    content: string;
    wordCount: number;
    applied: boolean;
    tokenUsage?: ChapterReviewCycleUsage;
  }>;
  readonly assertChapterContentNotEmpty: (content: string, stage: string) => void;
  readonly addUsage: (
    left: ChapterReviewCycleUsage,
    right?: ChapterReviewCycleUsage,
  ) => ChapterReviewCycleUsage;
  readonly restoreAssessment: (
    previous: ChapterAssessment,
    next: ChapterAssessment,
  ) => ChapterAssessment;
  readonly logWarn: (message: { zh: string; en: string }) => void;
  readonly logStage: (message: { zh: string; en: string }) => void;
}): Promise<ChapterReviewCycleResult> {
  const assess = async (
    chapterContent: string,
    options?: {
      temperature?: number;
      initialRepairIssues?: ReadonlyArray<AuditIssue>;
    },
  ): Promise<ChapterAssessment> => {
    const assessment = await params.assessChapter(chapterContent, options);
    totalUsage = params.addUsage(totalUsage, assessment.auditResult.tokenUsage);
    return assessment;
  };

  let totalUsage = params.initialUsage;
  let postReviseCount = 0;
  let normalizeApplied = false;
  let finalContent = params.initialOutput.content;
  let finalWordCount = params.initialOutput.wordCount;
  let revised = false;

  const normalizedBeforeAudit = await params.normalizeDraftLengthIfNeeded(finalContent);
  totalUsage = params.addUsage(totalUsage, normalizedBeforeAudit.tokenUsage);
  finalContent = normalizedBeforeAudit.content;
  finalWordCount = normalizedBeforeAudit.wordCount;
  normalizeApplied = normalizeApplied || normalizedBeforeAudit.applied;
  params.assertChapterContentNotEmpty(finalContent, "draft generation");

  if ((params.initialRepairIssues?.length ?? 0) > 0) {
    params.logWarn({
      zh: `首轮评审接收了 ${params.initialRepairIssues!.length} 条预检修复问题`,
      en: `${params.initialRepairIssues!.length} preflight repair issues were fed into the first assessment`,
    });
  }

  params.logStage({ zh: "审计草稿", en: "auditing draft" });
  let assessment = await assess(finalContent, {
    initialRepairIssues: params.initialRepairIssues ?? [],
  });

  while (assessment.repairDecision !== "none" && assessment.repairIssues.length > 0) {
    const repairMode = assessment.repairDecision;
    params.logStage(
      repairMode === "local-fix"
        ? { zh: "自动修复当前章的局部问题", en: "auto-fixing local issues in the current chapter" }
        : { zh: "当前章局部修复未通过，升级为整章改写", en: "local repair still failed, escalating to full chapter rewrite" },
    );
    const reviseOutput = await params.repairChapter(
      finalContent,
      assessment.repairIssues,
      repairMode,
    );
    totalUsage = params.addUsage(totalUsage, reviseOutput.tokenUsage);

    if (reviseOutput.revisedContent.length === 0 || reviseOutput.revisedContent === finalContent) {
      if (repairMode === "rewrite") {
        break;
      }
      continue;
    }

    const normalizedRevision = await params.normalizeDraftLengthIfNeeded(reviseOutput.revisedContent);
    totalUsage = params.addUsage(totalUsage, normalizedRevision.tokenUsage);
    postReviseCount = normalizedRevision.wordCount;
    normalizeApplied = normalizeApplied || normalizedRevision.applied;
    const previousAssessment = assessment;
    const previousContent = finalContent;

    const nextAssessment = params.restoreAssessment(
      previousAssessment,
      await assess(normalizedRevision.content, { temperature: 0 }),
    );
    if (nextAssessment.aiTellCount > previousAssessment.aiTellCount) {
      assessment = params.restoreAssessment(
        previousAssessment,
        await assess(previousContent, { temperature: 0 }),
      );
      break;
    }

    finalContent = normalizedRevision.content;
    finalWordCount = normalizedRevision.wordCount;
    revised = true;
    params.assertChapterContentNotEmpty(finalContent, repairMode === "local-fix" ? "revision" : "rewrite");
    assessment = nextAssessment;
  }

  return {
    finalContent,
    finalWordCount,
    preAuditNormalizedWordCount: normalizedBeforeAudit.wordCount,
    revised,
    auditResult: assessment.auditResult,
    totalUsage,
    postReviseCount,
    normalizeApplied,
  };
}
