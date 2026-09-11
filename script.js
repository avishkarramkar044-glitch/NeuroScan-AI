document.addEventListener("DOMContentLoaded", function () {

    // =========================================================
    // 2D / 3D MODEL FAMILY SWITCHING
    // =========================================================

    const fileInput = document.getElementById("file-input");
    const fileName = document.getElementById("file-name");
    const fileHelp = document.getElementById("file-help");
    const uploadIcon = document.getElementById("upload-icon");
    const uploadArea = document.getElementById("upload-area");
    const previewContainer = document.getElementById("preview-container");
    const imagePreview = document.getElementById("image-preview");
    const volumePreviewNote = document.getElementById("volume-preview-note");

    const architecture2d = document.getElementById("architecture-2d");
    const architecture3d = document.getElementById("architecture-3d");
    const modelType2d = document.getElementById("model-type-2d");
    const modelType3d = document.getElementById("model-type-3d");

    const family2dCard = document.getElementById("family-2d-card");
    const family3dCard = document.getElementById("family-3d-card");


    // =========================================================
    // GET SELECTED MODEL FAMILY
    // =========================================================

    function getFamily() {

        const selected = document.querySelector(
            'input[name="model_family"]:checked'
        );

        return selected ? selected.value : "2d";
    }


    // =========================================================
    // SET MODEL FAMILY
    // =========================================================

    window.setModelFamily = function (family) {

        const is3d = family === "3d";


        // -----------------------------------------------------
        // Show / hide architecture panels
        // -----------------------------------------------------

        if (architecture2d) {
            architecture2d.classList.toggle("hidden", is3d);
        }

        if (architecture3d) {
            architecture3d.classList.toggle("hidden", !is3d);
        }


        // -----------------------------------------------------
        // Enable only selected model dropdown
        // -----------------------------------------------------

        if (modelType2d) {
            modelType2d.disabled = is3d;
        }

        if (modelType3d) {
            modelType3d.disabled = !is3d;
        }


        // -----------------------------------------------------
        // Highlight selected family
        // -----------------------------------------------------

        if (family2dCard) {

            family2dCard.classList.toggle(
                "ring-2",
                !is3d
            );

        }

        if (family3dCard) {

            family3dCard.classList.toggle(
                "ring-2",
                is3d
            );

        }


        // -----------------------------------------------------
        // Change accepted file type
        // -----------------------------------------------------

        if (fileInput) {

            if (is3d) {

                fileInput.accept =
                    ".nii,.nii.gz,application/gzip,application/octet-stream";

            } else {

                fileInput.accept =
                    ".png,.jpg,.jpeg,.webp,image/png,image/jpeg,image/webp";

            }

            // Clear previous file when changing pipeline
            fileInput.value = "";
        }


        // -----------------------------------------------------
        // Change upload text
        // -----------------------------------------------------

        if (fileName) {

            fileName.textContent = is3d
                ? "Choose a 3D NIfTI volume"
                : "Choose an MRI image";

        }


        if (fileHelp) {

            fileHelp.textContent = is3d
                ? "Click or drag a .nii / .nii.gz volume here"
                : "Click or drag a 2D MRI image here";

        }


        // -----------------------------------------------------
        // Change upload icon
        // -----------------------------------------------------

        if (uploadIcon) {

            uploadIcon.className = is3d
                ? "fa-solid fa-cubes text-4xl text-violet-300"
                : "fa-solid fa-image text-4xl text-violet-300";

        }


        // -----------------------------------------------------
        // Reset preview
        // -----------------------------------------------------

        if (previewContainer) {
            previewContainer.classList.add("hidden");
        }

        if (imagePreview) {

            imagePreview.classList.remove("hidden");
            imagePreview.removeAttribute("src");

        }

        if (volumePreviewNote) {
            volumePreviewNote.classList.add("hidden");
        }

    };


    // =========================================================
    // SHOW SELECTED FILE
    // =========================================================

    function showSelectedFile(file) {

        if (!file) {
            return;
        }


        const is3d = getFamily() === "3d";


        // -----------------------------------------------------
        // Display filename
        // -----------------------------------------------------

        if (fileName) {
            fileName.textContent = file.name;
        }


        // -----------------------------------------------------
        // 3D FILE
        // -----------------------------------------------------

        if (is3d) {

            if (previewContainer) {
                previewContainer.classList.remove("hidden");
            }

            if (imagePreview) {
                imagePreview.classList.add("hidden");
            }

            if (volumePreviewNote) {
                volumePreviewNote.classList.remove("hidden");
            }

            return;
        }


        // -----------------------------------------------------
        // 2D IMAGE
        // -----------------------------------------------------

        if (
            imagePreview &&
            file.type &&
            file.type.startsWith("image/")
        ) {

            imagePreview.src =
                URL.createObjectURL(file);

            imagePreview.classList.remove("hidden");

            if (volumePreviewNote) {
                volumePreviewNote.classList.add("hidden");
            }

            if (previewContainer) {
                previewContainer.classList.remove("hidden");
            }

        }

    }


    // =========================================================
    // FILE INPUT CHANGE
    // =========================================================

    if (fileInput) {

        fileInput.addEventListener(
            "change",
            function () {

                showSelectedFile(
                    this.files?.[0]
                );

            }
        );

    }


    // =========================================================
    // MODEL FAMILY RADIO BUTTONS
    // =========================================================

    document
        .querySelectorAll(
            'input[name="model_family"]'
        )
        .forEach(function (radio) {

            radio.addEventListener(
                "change",
                function () {

                    window.setModelFamily(
                        this.value
                    );

                }
            );

        });


    // =========================================================
    // INITIAL MODEL FAMILY STATE
    // =========================================================

    window.setModelFamily(
        getFamily()
    );


    // =========================================================
    // DRAG AND DROP
    // =========================================================

    if (
        uploadArea &&
        fileInput
    ) {

        // -----------------------------------------------------
        // Drag enter / drag over
        // -----------------------------------------------------

        [
            "dragenter",
            "dragover"
        ].forEach(function (eventName) {

            uploadArea.addEventListener(
                eventName,
                function (event) {

                    event.preventDefault();
                    event.stopPropagation();

                    uploadArea.classList.add(
                        "border-cyan-400",
                        "bg-cyan-400/5"
                    );

                }
            );

        });


        // -----------------------------------------------------
        // Drag leave / drop
        // -----------------------------------------------------

        [
            "dragleave",
            "drop"
        ].forEach(function (eventName) {

            uploadArea.addEventListener(
                eventName,
                function (event) {

                    event.preventDefault();
                    event.stopPropagation();

                    uploadArea.classList.remove(
                        "border-cyan-400",
                        "bg-cyan-400/5"
                    );

                }
            );

        });


        // -----------------------------------------------------
        // Drop file
        // -----------------------------------------------------

        uploadArea.addEventListener(
            "drop",
            function (event) {

                const files =
                    event.dataTransfer.files;

                if (
                    files &&
                    files.length > 0
                ) {

                    try {

                        fileInput.files =
                            files;

                    } catch (error) {

                        console.warn(
                            "Browser prevented assigning dropped files."
                        );

                    }

                    showSelectedFile(
                        files[0]
                    );

                }

            }
        );

    }


    // =========================================================
    // ANALYSIS LOADING
    // =========================================================

    const predictionForm =
        document.getElementById(
            "analysis-form"
        );


    if (predictionForm) {

        predictionForm.addEventListener(
            "submit",
            function () {

                const button =
                    document.getElementById(
                        "analyze-button"
                    );


                if (button) {

                    button.disabled = true;

                    button.innerHTML =
                        '<i class="fa-solid fa-spinner fa-spin mr-2"></i>' +
                        "Analyzing selected pipeline...";

                }

            }
        );

    }


    // =========================================================
    // IMAGE MODAL
    // =========================================================

    const viewers =
        document.querySelectorAll(
            ".explainability-viewer"
        );


    viewers.forEach(function (viewer) {

        viewer.addEventListener(
            "click",
            function () {

                const source =
                    this.dataset.image;

                const title =
                    this.dataset.title ||
                    "Explainability image";


                const modal =
                    document.getElementById(
                        "image-modal"
                    );

                const modalImage =
                    document.getElementById(
                        "modal-image"
                    );

                const modalTitle =
                    document.getElementById(
                        "modal-title"
                    );


                if (
                    modal &&
                    modalImage &&
                    source
                ) {

                    modalImage.src =
                        source;


                    if (modalTitle) {

                        modalTitle.textContent =
                            title;

                    }


                    modal.classList.remove(
                        "hidden"
                    );

                }

            }
        );

    });


    // =========================================================
    // CLOSE IMAGE MODAL
    // =========================================================

    const closeModal =
        document.getElementById(
            "close-image-modal"
        );


    if (closeModal) {

        closeModal.addEventListener(
            "click",
            function () {

                document
                    .getElementById(
                        "image-modal"
                    )
                    ?.classList.add(
                        "hidden"
                    );

            }
        );

    }


    // =========================================================
    // ESCAPE KEY CLOSES MODAL
    // =========================================================

    document.addEventListener(
        "keydown",
        function (event) {

            if (event.key === "Escape") {

                document
                    .getElementById(
                        "image-modal"
                    )
                    ?.classList.add(
                        "hidden"
                    );

            }

        }
    );


    // =========================================================
    // COPY JSON
    // =========================================================

    const jsonButton =
        document.getElementById(
            "copy-json"
        );


    if (jsonButton) {

        jsonButton.addEventListener(
            "click",
            async function () {

                const jsonData =
                    document.getElementById(
                        "json-data"
                    );


                if (!jsonData) {
                    return;
                }


                try {

                    await navigator.clipboard.writeText(
                        jsonData.textContent
                    );


                    const oldText =
                        jsonButton.innerHTML;


                    jsonButton.innerHTML =
                        '<i class="fa-solid fa-check mr-2"></i>Copied';


                    setTimeout(
                        function () {

                            jsonButton.innerHTML =
                                oldText;

                        },
                        1500
                    );


                } catch (error) {

                    console.error(
                        "Could not copy JSON:",
                        error
                    );

                }

            }
        );

    }


    // =========================================================
    // PROBABILITY BARS
    // =========================================================

    document
        .querySelectorAll(
            ".probability-bar"
        )
        .forEach(function (bar) {

            const width =
                bar.dataset.width ||
                "0";


            bar.style.width =
                "0%";


            setTimeout(
                function () {

                    bar.style.width =
                        width + "%";

                },
                150
            );

        });


    // =========================================================
    // SMOOTH SCROLL
    // =========================================================

    document
        .querySelectorAll(
            'a[href^="#"]'
        )
        .forEach(function (link) {

            link.addEventListener(
                "click",
                function (event) {

                    const target =
                        document.querySelector(
                            this.getAttribute(
                                "href"
                            )
                        );


                    if (target) {

                        event.preventDefault();

                        target.scrollIntoView({
                            behavior: "smooth"
                        });

                    }

                }
            );

        });

});