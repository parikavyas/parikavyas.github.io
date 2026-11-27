import "server-only";
import { wonderTopics } from "@/data/wonder-topics";
import { learningTopics } from "@/data/learning-topics";
import { showDrafts } from "@/lib/utils";

/** Topics a visitor can see (drafts only in local preview). */
export const visibleWonderTopics = () => wonderTopics.filter((t) => showDrafts || t.status === "published");
export const visibleLearningTopics = () => learningTopics.filter((t) => showDrafts || !t.draft);
