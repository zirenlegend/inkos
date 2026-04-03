import { describe, expect, it, vi } from "vitest";
import { runChapterReviewCycle, type ChapterAssessment } from "../pipeline/chapter-review-cycle.js";
import type { AuditResult, AuditIssue } from "../agents/continuity.js";
import type { LengthSpec } from "../models/length-governance.js";

const LENGTH_SPEC: LengthSpec = {
  target: 220,
  softMin: 190,
  softMax: 250,
  hardMin: 160,
  hardMax: 280,
  countingMode: "zh_chars",
  normalizeMode: "none",
};

const ZERO_USAGE = {
  promptTokens: 0,
  completionTokens: 0,
  totalTokens: 0,
} as const;

function createAuditResult(overrides?: Partial<AuditResult>): AuditResult {
  return {
    passed: true,
    issues: [],
    summary: "clean",
    ...overrides,
  };
}

function createAssessment(overrides?: Partial<ChapterAssessment>): ChapterAssessment {
  return {
    auditResult: createAuditResult(),
    repairIssues: [],
    repairDecision: "none",
    aiTellCount: 0,
    blockingCount: 0,
    criticalCount: 0,
    ...overrides,
  };
}

describe("runChapterReviewCycle", () => {
  it("lets the initial assessment choose rewrite instead of forcing local-fix", async () => {
    const initialRepairIssues: AuditIssue[] = [{
      severity: "critical",
      category: "paragraph-shape",
      description: "too fragmented",
      suggestion: "merge short fragments",
    }];
    const assessChapter = vi.fn()
      .mockResolvedValueOnce(createAssessment({
        auditResult: createAuditResult({
          passed: false,
          issues: initialRepairIssues,
          summary: "rewrite directly",
        }),
        repairIssues: initialRepairIssues,
        repairDecision: "rewrite",
        blockingCount: 1,
        criticalCount: 1,
      }))
      .mockResolvedValueOnce(createAssessment({
        auditResult: createAuditResult({
          passed: true,
          issues: [],
          summary: "clean",
        }),
      }));
    const reviseChapter = vi.fn().mockResolvedValue({
      revisedContent: "rewritten draft",
      wordCount: 10,
      fixedIssues: ["fixed"],
      updatedState: "",
      updatedLedger: "",
      updatedHooks: "",
      tokenUsage: ZERO_USAGE,
    });
    const normalizeDraftLengthIfNeeded = vi.fn()
      .mockResolvedValueOnce({
        content: "raw draft",
        wordCount: 9,
        applied: false,
        tokenUsage: ZERO_USAGE,
      })
      .mockResolvedValueOnce({
        content: "rewritten draft",
        wordCount: 10,
        applied: false,
        tokenUsage: ZERO_USAGE,
      });

    const result = await runChapterReviewCycle({
      initialOutput: { content: "raw draft", wordCount: 9 },
      initialRepairIssues,
      lengthSpec: LENGTH_SPEC,
      initialUsage: ZERO_USAGE,
      assessChapter,
      repairChapter: (chapterContent, issues, mode) =>
        reviseChapter("/tmp/book", chapterContent, 1, issues, mode, "xuanhuan", {
          lengthSpec: LENGTH_SPEC,
        }),
      normalizeDraftLengthIfNeeded,
      assertChapterContentNotEmpty: () => undefined,
      addUsage: (left, right) => ({
        promptTokens: left.promptTokens + (right?.promptTokens ?? 0),
        completionTokens: left.completionTokens + (right?.completionTokens ?? 0),
        totalTokens: left.totalTokens + (right?.totalTokens ?? 0),
      }),
      restoreAssessment: (_previous, next) => next,
      logWarn: () => undefined,
      logStage: () => undefined,
    });

    expect(reviseChapter).toHaveBeenCalledTimes(1);
    expect(reviseChapter).toHaveBeenCalledWith(
      "/tmp/book",
      "raw draft",
      1,
      initialRepairIssues,
      "rewrite",
      "xuanhuan",
      expect.any(Object),
    );
    expect(assessChapter).toHaveBeenNthCalledWith(1, "raw draft", {
      initialRepairIssues,
    });
    expect(result.finalContent).toBe("rewritten draft");
    expect(result.revised).toBe(true);
  });

  it("drops auto-revision when it increases AI tells and re-audits the original draft", async () => {
    const failingIssues: AuditIssue[] = [{
      severity: "warning",
      category: "continuity",
      description: "broken continuity",
      suggestion: "fix it",
    }];
    const assessChapter = vi.fn()
      .mockResolvedValueOnce(createAssessment({
        auditResult: createAuditResult({
          passed: false,
          issues: failingIssues,
          summary: "bad",
        }),
        repairIssues: failingIssues,
        repairDecision: "local-fix",
        blockingCount: 1,
        criticalCount: 0,
        aiTellCount: 0,
      }))
      .mockResolvedValueOnce(createAssessment({
        auditResult: createAuditResult({
          passed: false,
          issues: failingIssues,
          summary: "still noisy",
        }),
        repairIssues: failingIssues,
        repairDecision: "local-fix",
        blockingCount: 1,
        criticalCount: 0,
        aiTellCount: 1,
      }))
      .mockResolvedValueOnce(createAssessment({
        auditResult: createAuditResult({
          passed: false,
          issues: failingIssues,
          summary: "fallback original",
        }),
        repairIssues: failingIssues,
        repairDecision: "local-fix",
        blockingCount: 1,
        criticalCount: 0,
        aiTellCount: 0,
      }));
    const reviseChapter = vi.fn().mockResolvedValue({
      revisedContent: "rewritten draft",
      wordCount: 15,
      fixedIssues: ["fixed"],
      updatedState: "",
      updatedLedger: "",
      updatedHooks: "",
      tokenUsage: ZERO_USAGE,
    });
    const normalizeDraftLengthIfNeeded = vi.fn()
      .mockResolvedValueOnce({
        content: "original draft",
        wordCount: 13,
        applied: false,
        tokenUsage: ZERO_USAGE,
      })
      .mockResolvedValueOnce({
        content: "rewritten draft",
        wordCount: 15,
        applied: false,
        tokenUsage: ZERO_USAGE,
      });
    const analyzeAITells = vi.fn((content: string) => ({
      issues: content === "rewritten draft"
        ? [{ severity: "warning", category: "ai", description: "more ai", suggestion: "reduce" } satisfies AuditIssue]
        : [],
    }));

    const result = await runChapterReviewCycle({
      initialOutput: {
        content: "original draft",
        wordCount: 13,
      },
      initialRepairIssues: [],
      lengthSpec: LENGTH_SPEC,
      initialUsage: ZERO_USAGE,
      assessChapter,
      repairChapter: (chapterContent, issues, mode) =>
        reviseChapter("/tmp/book", chapterContent, 1, issues, mode, "xuanhuan", {
          lengthSpec: LENGTH_SPEC,
        }),
      normalizeDraftLengthIfNeeded,
      assertChapterContentNotEmpty: () => undefined,
      addUsage: (left, right) => ({
        promptTokens: left.promptTokens + (right?.promptTokens ?? 0),
        completionTokens: left.completionTokens + (right?.completionTokens ?? 0),
        totalTokens: left.totalTokens + (right?.totalTokens ?? 0),
      }),
      restoreAssessment: (_previous, next) => next,
      logWarn: () => undefined,
      logStage: () => undefined,
    });

    expect(reviseChapter).toHaveBeenCalledTimes(1);
    expect(assessChapter).toHaveBeenNthCalledWith(1, "original draft", { initialRepairIssues: [] });
    expect(assessChapter).toHaveBeenNthCalledWith(2, "rewritten draft", { temperature: 0 });
    expect(assessChapter).toHaveBeenNthCalledWith(3, "original draft", { temperature: 0 });
    expect(result.finalContent).toBe("original draft");
    expect(result.revised).toBe(false);
  });

  it("follows fresh repair decisions from each assessment round", async () => {
    const firstIssues: AuditIssue[] = [{
      severity: "warning",
      category: "dialogue",
      description: "line reads flat",
      suggestion: "tighten the exchange",
    }];
    const secondIssues: AuditIssue[] = [{
      severity: "critical",
      category: "continuity",
      description: "chapter still breaks continuity",
      suggestion: "rewrite the scene flow",
    }];
    const assessChapter = vi.fn()
      .mockResolvedValueOnce(createAssessment({
        auditResult: createAuditResult({
          passed: false,
          issues: firstIssues,
          summary: "needs local repair",
        }),
        repairIssues: firstIssues,
        repairDecision: "local-fix",
        blockingCount: 1,
        criticalCount: 0,
      }))
      .mockResolvedValueOnce(createAssessment({
        auditResult: createAuditResult({
          passed: false,
          issues: secondIssues,
          summary: "rewrite now",
        }),
        repairIssues: secondIssues,
        repairDecision: "rewrite",
        blockingCount: 1,
        criticalCount: 1,
      }))
      .mockResolvedValueOnce(createAssessment({
        auditResult: createAuditResult({
          passed: true,
          issues: [],
          summary: "clean",
        }),
      }));
    const reviseChapter = vi.fn()
      .mockResolvedValueOnce({
        revisedContent: "locally fixed draft",
        wordCount: 16,
        fixedIssues: ["fixed local detail"],
        updatedState: "",
        updatedLedger: "",
        updatedHooks: "",
        tokenUsage: ZERO_USAGE,
      })
      .mockResolvedValueOnce({
        revisedContent: "fully rewritten draft",
        wordCount: 18,
        fixedIssues: ["rewrote chapter"],
        updatedState: "",
        updatedLedger: "",
        updatedHooks: "",
        tokenUsage: ZERO_USAGE,
      });
    const normalizeDraftLengthIfNeeded = vi.fn()
      .mockResolvedValueOnce({
        content: "original draft",
        wordCount: 13,
        applied: false,
        tokenUsage: ZERO_USAGE,
      })
      .mockResolvedValueOnce({
        content: "locally fixed draft",
        wordCount: 16,
        applied: false,
        tokenUsage: ZERO_USAGE,
      })
      .mockResolvedValueOnce({
        content: "fully rewritten draft",
        wordCount: 18,
        applied: false,
        tokenUsage: ZERO_USAGE,
      });

    const result = await runChapterReviewCycle({
      initialOutput: { content: "original draft", wordCount: 13 },
      initialRepairIssues: [],
      lengthSpec: LENGTH_SPEC,
      initialUsage: ZERO_USAGE,
      assessChapter,
      repairChapter: (chapterContent, issues, mode) =>
        reviseChapter("/tmp/book", chapterContent, 1, issues, mode, "xuanhuan", {
          lengthSpec: LENGTH_SPEC,
        }),
      normalizeDraftLengthIfNeeded,
      assertChapterContentNotEmpty: () => undefined,
      addUsage: (left, right) => ({
        promptTokens: left.promptTokens + (right?.promptTokens ?? 0),
        completionTokens: left.completionTokens + (right?.completionTokens ?? 0),
        totalTokens: left.totalTokens + (right?.totalTokens ?? 0),
      }),
      restoreAssessment: (_previous, next) => next,
      logWarn: () => undefined,
      logStage: () => undefined,
    });

    expect(reviseChapter).toHaveBeenNthCalledWith(
      1,
      "/tmp/book",
      "original draft",
      1,
      firstIssues,
      "local-fix",
      "xuanhuan",
      expect.any(Object),
    );
    expect(reviseChapter).toHaveBeenNthCalledWith(
      2,
      "/tmp/book",
      "locally fixed draft",
      1,
      secondIssues,
      "rewrite",
      "xuanhuan",
      expect.any(Object),
    );
    expect(assessChapter).toHaveBeenNthCalledWith(1, "original draft", { initialRepairIssues: [] });
    expect(assessChapter).toHaveBeenNthCalledWith(2, "locally fixed draft", { temperature: 0 });
    expect(assessChapter).toHaveBeenNthCalledWith(3, "fully rewritten draft", { temperature: 0 });
    expect(result.finalContent).toBe("fully rewritten draft");
    expect(result.revised).toBe(true);
    expect(result.auditResult.passed).toBe(true);
  });
});
