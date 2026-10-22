const input = document.getElementById("photo");
const preview = document.getElementById("preview");
const statusEl = document.getElementById("status");
const captionEl = document.getElementById("caption");
const repeatBtn = document.getElementById("repeat");

function speak(text) {
  if (!("speechSynthesis" in window)) return;
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 0.95;
  window.speechSynthesis.speak(utterance);
}

input.addEventListener("change", async () => {
  const file = input.files[0];
  if (!file) return;
  preview.src = URL.createObjectURL(file);
  preview.hidden = false;
  captionEl.textContent = "";
  repeatBtn.hidden = true;
  statusEl.textContent = "Looking at the photo...";

  const form = new FormData();
  form.append("image", file);
  try {
    const res = await fetch("/api/caption", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Something went wrong.");
    captionEl.textContent = data.caption;
    preview.alt = data.caption;
    statusEl.textContent = `Described in ${(data.latency_ms / 1000).toFixed(1)} seconds.`;
    repeatBtn.hidden = false;
    speak(data.caption);
  } catch (err) {
    statusEl.textContent = err.message;
    speak(err.message);
  }
});

repeatBtn.addEventListener("click", () => speak(captionEl.textContent));
