"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

const STORAGE_KEY = "prism-api-key";

interface ApiKeyContextValue {
  apiKey: string;
  ready: boolean;
  setApiKey: (value: string) => void;
  clearApiKey: () => void;
}

const ApiKeyContext = createContext<ApiKeyContextValue | null>(null);

export function ApiKeyProvider({ children }: { children: ReactNode }) {
  const [apiKey, setApiKeyState] = useState("");
  const [ready, setReady] = useState(false);

  useEffect(() => {
    try {
      setApiKeyState(window.localStorage.getItem(STORAGE_KEY) ?? "");
    } catch {
      setApiKeyState("");
    }
    setReady(true);
  }, []);

  const setApiKey = useCallback((value: string) => {
    const cleaned = value.trim();
    setApiKeyState(cleaned);
    try {
      if (cleaned) {
        window.localStorage.setItem(STORAGE_KEY, cleaned);
      } else {
        window.localStorage.removeItem(STORAGE_KEY);
      }
    } catch {
      // Storage can be unavailable (private mode); the in-memory key still works.
    }
  }, []);

  const clearApiKey = useCallback(() => setApiKey(""), [setApiKey]);

  const value = useMemo(
    () => ({ apiKey, ready, setApiKey, clearApiKey }),
    [apiKey, ready, setApiKey, clearApiKey],
  );

  return <ApiKeyContext.Provider value={value}>{children}</ApiKeyContext.Provider>;
}

export function useApiKey(): ApiKeyContextValue {
  const context = useContext(ApiKeyContext);
  if (!context) {
    throw new Error("useApiKey must be used within ApiKeyProvider");
  }
  return context;
}
