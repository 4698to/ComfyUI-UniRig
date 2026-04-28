import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "UniRig.LoadMeshDynamicFiles",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== "UniRigLoadMesh") {
            return;
        }

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const ret = origOnNodeCreated ? origOnNodeCreated.apply(this, arguments) : undefined;

            const sourceWidget = this.widgets?.find((w) => w.name === "source_folder");
            const filePathWidget = this.widgets?.find((w) => w.name === "file_path");
            if (!sourceWidget || !filePathWidget) {
                return ret;
            }

            let selectorWidget =
                this.widgets?.find((w) => w.name === "mesh_selector") ||
                this.addWidget("combo", "mesh_selector", "", () => {}, { values: ["No mesh files found"] });

            const refreshFileOptions = async () => {
                const source = sourceWidget.value || "input";
                try {
                    // Use native fetch with route fallback.
                    let resp = await fetch(`/unirig/mesh-files?source_folder=${encodeURIComponent(source)}`);
                    if (!resp.ok) return;

                    const data = await resp.json();
                    const files = Array.isArray(data?.files) && data.files.length > 0 ? data.files : ["No mesh files found"];

                    selectorWidget.options = selectorWidget.options || {};
                    selectorWidget.options.values = files;

                    if (!files.includes(selectorWidget.value)) {
                        selectorWidget.value = files[0];
                    }
                    if (!filePathWidget.value || !files.includes(filePathWidget.value)) {
                        filePathWidget.value = selectorWidget.value;
                    }

                    this.setDirtyCanvas(true, true);
                } catch (err) {
                    console.warn("[UniRig] failed to refresh mesh files:", err);
                }
            };

            const origSelectorCallback = selectorWidget.callback;
            selectorWidget.callback = (...args) => {
                if (origSelectorCallback) {
                    origSelectorCallback.apply(selectorWidget, args);
                }
                if (filePathWidget) {
                    filePathWidget.value = selectorWidget.value || "";
                }
            };

            const origSourceCallback = sourceWidget.callback;
            sourceWidget.callback = (...args) => {
                if (origSourceCallback) {
                    origSourceCallback.apply(sourceWidget, args);
                }
                refreshFileOptions();
            };

            refreshFileOptions();
            return ret;
        };
    },
});
