/**
 * pythonCode.ts
 * -------------
 * Imports the actual Python project files (kept at the repository root in
 * Traffic-Sign-Classification/) so the web app can display and offer them for
 * download — a single source of truth, no duplicated code.
 */
import trainPy from "../../Traffic-Sign-Classification/train.py?raw";
import trainPyUrl from "../../Traffic-Sign-Classification/train.py?url";
import predictPy from "../../Traffic-Sign-Classification/predict.py?raw";
import predictPyUrl from "../../Traffic-Sign-Classification/predict.py?url";
import appPy from "../../Traffic-Sign-Classification/app.py?raw";
import appPyUrl from "../../Traffic-Sign-Classification/app.py?url";
import signClassesPy from "../../Traffic-Sign-Classification/sign_classes.py?raw";
import signClassesPyUrl from "../../Traffic-Sign-Classification/sign_classes.py?url";
import reqTxt from "../../Traffic-Sign-Classification/requirements.txt?raw";
import reqTxtUrl from "../../Traffic-Sign-Classification/requirements.txt?url";
import readmeMd from "../../Traffic-Sign-Classification/README.md?raw";
import readmeMdUrl from "../../Traffic-Sign-Classification/README.md?url";

export interface CodeFile {
  name: string;
  path: string;
  lang: "python" | "markdown" | "text";
  code: string;
  url: string;
}

export const CODE_FILES: CodeFile[] = [
  { name: "train.py", path: "Traffic-Sign-Classification/train.py", lang: "python", code: trainPy, url: trainPyUrl },
  { name: "predict.py", path: "Traffic-Sign-Classification/predict.py", lang: "python", code: predictPy, url: predictPyUrl },
  { name: "app.py", path: "Traffic-Sign-Classification/app.py", lang: "python", code: appPy, url: appPyUrl },
  { name: "sign_classes.py", path: "Traffic-Sign-Classification/sign_classes.py", lang: "python", code: signClassesPy, url: signClassesPyUrl },
  { name: "requirements.txt", path: "Traffic-Sign-Classification/requirements.txt", lang: "text", code: reqTxt, url: reqTxtUrl },
  { name: "README.md", path: "Traffic-Sign-Classification/README.md", lang: "markdown", code: readmeMd, url: readmeMdUrl },
];
