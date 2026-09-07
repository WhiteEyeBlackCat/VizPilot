import type {
  ChartSpec,
  DatasetMeta,
  DatasetProfile,
  RecommendationsResponse,
  RenderResult,
} from "./types";

export class ApiError extends Error {
  constructor(
    public status: number,
    public errors: string[],
  ) {
    super(errors.join("; "));
  }
}

/** Normalizes both error shapes: the unified {detail: {errors: [...]}} 422
 *  and FastAPI's native detail (string or validation list). */
function extractErrors(body: unknown, status: number): string[] {
  const detail = body && typeof body === "object" ? (body as { detail?: unknown }).detail : null;
  if (detail && typeof detail === "object" && !Array.isArray(detail)) {
    const errors = (detail as { errors?: unknown }).errors;
    if (Array.isArray(errors)) return errors.map(String);
  }
  if (typeof detail === "string") return [detail];
  if (Array.isArray(detail)) {
    return detail.map((item) =>
      item && typeof item === "object" && "msg" in item
        ? String((item as { msg: unknown }).msg)
        : JSON.stringify(item),
    );
  }
  return [`HTTP ${status}`];
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(path, init);
  let body: unknown = null;
  try {
    body = await resp.json();
  } catch {
    // non-JSON body (proxy errors etc.) — handled below
  }
  if (!resp.ok) throw new ApiError(resp.status, extractErrors(body, resp.status));
  return body as T;
}

export const api = {
  upload(file: File): Promise<DatasetMeta> {
    const form = new FormData();
    form.append("file", file);
    return request("/api/datasets", { method: "POST", body: form });
  },
  listDatasets(): Promise<DatasetMeta[]> {
    return request("/api/datasets");
  },
  profile(datasetId: string): Promise<DatasetProfile> {
    return request(`/api/datasets/${datasetId}/profile`);
  },
  recommendations(datasetId: string, llm: boolean): Promise<RecommendationsResponse> {
    return request(`/api/datasets/${datasetId}/recommendations?llm=${llm}`);
  },
  render(datasetId: string, spec: ChartSpec): Promise<RenderResult> {
    return request("/api/charts/render", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset_id: datasetId, spec }),
    });
  },
};
