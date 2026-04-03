# =============================================================
#  AnalizeIA – Sistema Inteligente de Análise de Imagens com IA
#  Atividade Integradora – Pipeline Completo (Etapas 1-7) + Desafios 1-4
# =============================================================

import cv2
import numpy as np
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time

# ──────────────────────────────────────────────────────────────
# PALETA – Tema Azul Ciano Escuro
# ──────────────────────────────────────────────────────────────
C = {
    "bg":       "#0A1929",   # Fundo principal (azul marinho escuro)
    "panel":    "#0D2137",   # Painéis / sidebar (azul marinho médio)
    "card":     "#132F4C",   # Cards / campos (azul petróleo)
    "border":   "#1E4976",   # Bordas (azul ciano escuro)
    "accent":   "#5EEAD4",   # Acento principal (ciano vibrante)
    "accent2":  "#38BDF8",   # Acento secundário (azul céu)
    "ok":       "#34D399",   # Verde sucesso (verde esmeralda)
    "warn":     "#FBBF24",   # Amarelo alerta
    "danger":   "#F87171",   # Vermelho perigo (coral suave)
    "text":     "#E2E8F0",   # Texto principal (branco azulado)
    "sub":      "#94A3B8",   # Texto secundário (cinza azulado)
    "inactive": "#1E3A5F",   # Elementos inativos (azul escuro)
}

CONF_THRESHOLD  = 0.45
ALERT_THRESHOLD = 5
MODEL_PATH      = "yolov8n.pt"

TODAS_AS_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush",
]

# ──────────────────────────────────────────────────────────────
# HELPERS DE WIDGET
# ──────────────────────────────────────────────────────────────

def make_frame(parent, bg=None, border_color=None, bd=1, **kw):
    bg = bg or C["card"]
    if border_color:
        return tk.Frame(parent, bg=bg,
                        highlightbackground=border_color,
                        highlightthickness=bd, **kw)
    return tk.Frame(parent, bg=bg, **kw)

def make_label(parent, text, size=9, bold=False, color=None, bg=None, **kw):
    font = ("Segoe UI", size, "bold" if bold else "normal")
    return tk.Label(parent, text=text, font=font,
                    fg=color or C["text"], bg=bg or C["card"], **kw)

def make_btn(parent, text, color, fg_color, cmd):
    b = tk.Button(parent, text=text, command=cmd,
                  bg=color, fg=fg_color,
                  font=("Segoe UI", 9, "bold"),
                  relief="flat", padx=12, pady=6,
                  cursor="hand2", bd=0,
                  activebackground=color,
                  activeforeground=fg_color)
    def _enter(e): b.config(relief="groove")
    def _leave(e): b.config(relief="flat")
    b.bind("<Enter>", _enter)
    b.bind("<Leave>", _leave)
    return b

def sep(parent, color=None, padx=10, pady=6):
    tk.Frame(parent, bg=color or C["border"], height=1).pack(
        fill="x", padx=padx, pady=pady)

def style_notebook(nb: ttk.Notebook):
    """Aplica estilo escuro ao ttk.Notebook."""
    style = ttk.Style()
    style.theme_use("clam")
    style.configure("Dark.TNotebook",
                    background=C["bg"],
                    borderwidth=0)
    style.configure("Dark.TNotebook.Tab",
                    background=C["panel"],
                    foreground=C["sub"],
                    font=("Segoe UI", 9, "bold"),
                    padding=[14, 6])
    style.map("Dark.TNotebook.Tab",
              background=[("selected", C["card"])],
              foreground=[("selected", C["text"])])
    nb.configure(style="Dark.TNotebook")

# ──────────────────────────────────────────────────────────────
# PROCESSAMENTO – PIPELINE COMPLETO (Etapas 1–5)
# ──────────────────────────────────────────────────────────────

def processar_pipeline(frame: np.ndarray) -> dict:
    out = {}

    # Etapa 1 – Original
    out["original"] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Etapa 2 – Cinza / Blur / Canny
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 100, 200)
    out["gray"]  = cv2.cvtColor(gray,  cv2.COLOR_GRAY2RGB)
    out["blur"]  = cv2.cvtColor(blur,  cv2.COLOR_GRAY2RGB)
    out["edges"] = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    # Etapa 3 – HSV (matiz rotacionado + saturação amplificada) + canais
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    h_rot = (h + 90) % 180
    s_amp = np.clip(s * 1.4, 0, 255)
    hsv_mod = cv2.merge([h_rot, s_amp, v]).astype(np.uint8)
    out["hsv"] = cv2.cvtColor(cv2.cvtColor(hsv_mod, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2RGB)

    h8 = (h / 180 * 255).astype(np.uint8)
    s8 = s.astype(np.uint8)
    v8 = v.astype(np.uint8)
    out["canal_h"] = cv2.cvtColor(cv2.applyColorMap(h8, cv2.COLORMAP_HSV),  cv2.COLOR_BGR2RGB)
    out["canal_s"] = cv2.cvtColor(cv2.applyColorMap(s8, cv2.COLORMAP_BONE), cv2.COLOR_BGR2RGB)
    out["canal_v"] = cv2.cvtColor(cv2.applyColorMap(v8, cv2.COLORMAP_BONE), cv2.COLOR_BGR2RGB)

    # Etapa 4 – Histograma
    out["hist_data"] = gray.ravel()

    # Etapa 5 – Binarização Otsu
    thresh_val, thresh_img = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out["thresh"]     = cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2RGB)
    out["thresh_val"] = thresh_val

    return out

def thumb(img_rgb: np.ndarray, w=320, h=240) -> ImageTk.PhotoImage:
    return ImageTk.PhotoImage(Image.fromarray(cv2.resize(img_rgb, (w, h))))

# ──────────────────────────────────────────────────────────────
# FILTRO DE CLASSES
# ──────────────────────────────────────────────────────────────

class JanelaFiltro(tk.Toplevel):
    def __init__(self, parent, classes_ativas: set, callback):
        super().__init__(parent)
        self.title("AnalizeIA – Filtrar Classes")
        self.configure(bg=C["bg"])
        self.geometry("320x500")
        self.resizable(False, True)
        self.callback = callback

        make_label(self, "Selecionar Classes COCO", size=11, bold=True,
                   color=C["text"], bg=C["bg"]).pack(pady=(14, 2))
        make_label(self, "Somente os objetos marcados serão detectados.",
                   size=8, color=C["sub"], bg=C["bg"]).pack(pady=(0, 10))

        outer = tk.Frame(self, bg=C["bg"])
        outer.pack(fill="both", expand=True, padx=12)

        canvas = tk.Canvas(outer, bg=C["bg"], highlightthickness=0)
        scroll = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        inner  = tk.Frame(canvas, bg=C["bg"])
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        self._vars: dict[str, tk.BooleanVar] = {}
        for cls in sorted(TODAS_AS_CLASSES):
            v = tk.BooleanVar(value=(cls in classes_ativas))
            self._vars[cls] = v
            tk.Checkbutton(inner, text=f"  {cls}", variable=v,
                           bg=C["bg"], fg=C["text"],
                           selectcolor=C["card"],
                           font=("Segoe UI", 9),
                           activebackground=C["bg"],
                           activeforeground=C["accent"]).pack(anchor="w", pady=1)

        row = tk.Frame(self, bg=C["bg"])
        row.pack(pady=10)
        make_btn(row, "✔  Salvar",   C["ok"],     C["bg"], self._salvar).pack(side="left", padx=6)
        make_btn(row, "Cancelar",    C["card"],   C["text"], self.destroy).pack(side="left", padx=6)

    def _salvar(self):
        self.callback({c for c, v in self._vars.items() if v.get()})
        self.destroy()

# ──────────────────────────────────────────────────────────────
# APLICAÇÃO PRINCIPAL
# ──────────────────────────────────────────────────────────────

class AnalizeIA:

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("AnalizeIA – Sistema de Visão Computacional")
        self.root.configure(bg=C["bg"])
        self.root.resizable(True, True)

        # Estado
        self.running         = False
        self.cap             = None
        self.model           = None
        self.frame_count     = 0
        self.total_objetos   = 0
        self.t0              = 0.0
        self.modo_tracking   = tk.BooleanVar(value=False)
        self.filtro_ativo    = tk.BooleanVar(value=False)
        self.classes_filtro  : set[str] = {"person", "car", "dog"}
        self._ultimo_pipeline: dict | None = None
        self._tkimgs         : dict = {}   # evita GC dos thumbnails

        self._carregar_modelo()
        self._construir_ui()

    # ── Modelo ─────────────────────────────────────────────────
    def _carregar_modelo(self):
        try:
            self.model = YOLO(MODEL_PATH)
        except Exception as e:
            messagebox.showerror("AnalizeIA – Erro", str(e))

    # ── UI principal ───────────────────────────────────────────
    def _construir_ui(self):
        # ── Cabeçalho ──────────────────────────────────────────
        hdr = tk.Frame(self.root, bg=C["bg"])
        hdr.pack(fill="x", padx=16, pady=(12, 0))
        tk.Label(hdr, text="AnalizeIA",
                 bg=C["bg"], fg=C["text"],
                 font=("Segoe UI", 18, "bold")).pack(side="left")
        tk.Label(hdr, text="  Sistema Inteligente de Análise de Imagens com IA",
                 bg=C["bg"], fg=C["sub"],
                 font=("Segoe UI", 9)).pack(side="left", pady=(6, 0))
        tk.Frame(self.root, bg=C["border"], height=1).pack(fill="x", padx=16, pady=(8, 0))

        # ── Notebook (abas) ────────────────────────────────────
        self.nb = ttk.Notebook(self.root)
        style_notebook(self.nb)
        self.nb.pack(fill="both", expand=True, padx=16, pady=8)

        # ── Aba 1: Detecção IA ─────────────────────────────────
        aba_det = tk.Frame(self.nb, bg=C["bg"])
        self.nb.add(aba_det, text="  Detecção IA  ")
        self._construir_aba_deteccao(aba_det)

        # ── Aba 2: Pipeline ────────────────────────────────────
        aba_pip = tk.Frame(self.nb, bg=C["bg"])
        self.nb.add(aba_pip, text="  Pipeline (Etapas 1–5)  ")
        self._construir_aba_pipeline(aba_pip)

        # ── Rodapé ─────────────────────────────────────────────
        tk.Frame(self.root, bg=C["border"], height=1).pack(fill="x", padx=16)
        tk.Label(self.root,
                 text="AnalizeIA  ·  YOLOv8n  ·  OpenCV  ·  Atividade Integradora",
                 bg=C["bg"], fg=C["sub"], font=("Segoe UI", 7)).pack(pady=6)

    # ══════════════════════════════════════════════════════════
    # ABA 1 – DETECÇÃO IA
    # ══════════════════════════════════════════════════════════

    def _construir_aba_deteccao(self, parent):
        center = tk.Frame(parent, bg=C["bg"])
        center.pack(fill="both", expand=True, padx=10, pady=10)

        # ── Canvas de vídeo ────────────────────────────────────
        video_wrap = make_frame(center, bg=C["card"],
                                border_color=C["border"], bd=1)
        video_wrap.grid(row=0, column=0, padx=(0, 10), sticky="nsew")

        self.canvas_video = tk.Canvas(video_wrap, width=640, height=480,
                                      bg=C["inactive"], highlightthickness=0)
        self.canvas_video.pack(padx=2, pady=2)
        self._placeholder()

        # ── Sidebar ────────────────────────────────────────────
        side = tk.Frame(center, bg=C["panel"], width=255,
                        highlightbackground=C["border"],
                        highlightthickness=1)
        side.grid(row=0, column=1, sticky="nsew")
        side.grid_propagate(False)
        center.grid_columnconfigure(0, weight=1)

        # Título sidebar
        tk.Label(side, text="Dashboard", bg=C["panel"], fg=C["text"],
                 font=("Segoe UI", 12, "bold")).pack(pady=(14, 2))
        sep(side, C["border"])

        # Métricas
        def metric(titulo, attr, val_color=None):
            f = make_frame(side, bg=C["card"], border_color=C["border"], bd=1)
            f.pack(fill="x", padx=10, pady=4)
            tk.Label(f, text=titulo, bg=C["card"], fg=C["sub"],
                     font=("Segoe UI", 7)).pack(anchor="w", padx=8, pady=(6, 0))
            lbl = tk.Label(f, text="—", bg=C["card"],
                           fg=val_color or C["accent"],
                           font=("Segoe UI", 20, "bold"))
            lbl.pack(anchor="w", padx=8, pady=(0, 6))
            setattr(self, attr, lbl)

        metric("FPS",               "lbl_fps")
        metric("Objetos no frame",  "lbl_frame_objs")
        metric("Total acumulado",   "lbl_total")
        metric("Threshold Otsu",    "lbl_thresh", C["accent2"])

        sep(side, C["border"])

        self.lbl_status = tk.Label(side, text="● Aguardando",
                                   bg=C["panel"], fg=C["sub"],
                                   font=("Segoe UI", 10, "bold"))
        self.lbl_status.pack(pady=6)

        sep(side, C["border"])

        # Classes detectadas
        tk.Label(side, text="Classes detectadas:", bg=C["panel"], fg=C["sub"],
                 font=("Segoe UI", 7)).pack(anchor="w", padx=12, pady=(4, 2))

        self.txt_classes = tk.Text(side, bg=C["bg"], fg=C["ok"],
                                   font=("Consolas", 9), height=9,
                                   state="disabled", relief="flat",
                                   bd=0, padx=6)
        self.txt_classes.pack(padx=10, fill="x", pady=(0, 4))

        sep(side, C["border"])

        # Slider de confiança
        tk.Label(side, text="Confiança mínima:", bg=C["panel"], fg=C["sub"],
                 font=("Segoe UI", 7)).pack(anchor="w", padx=12, pady=(4, 0))

        self.slider_conf = tk.Scale(side, from_=0.1, to=0.95, resolution=0.05,
                                    orient="horizontal",
                                    bg=C["panel"], fg=C["text"],
                                    highlightthickness=0,
                                    troughcolor=C["inactive"],
                                    activebackground=C["accent"],
                                    length=220, font=("Segoe UI", 8))
        self.slider_conf.set(CONF_THRESHOLD)
        self.slider_conf.pack(padx=10, pady=(0, 8))

        # ── Barra de controles ──────────────────────────────────
        ctrl = tk.Frame(parent, bg=C["bg"])
        ctrl.pack(fill="x", padx=10, pady=(0, 8))

        btn_row = tk.Frame(ctrl, bg=C["bg"])
        btn_row.pack(side="left")

        make_btn(btn_row, "▶  Webcam",     C["ok"],     C["bg"],   self.iniciar_webcam).pack(side="left", padx=4)
        make_btn(btn_row, "📁  Imagem",    C["card"],   C["text"], self.abrir_imagem  ).pack(side="left", padx=4)
        make_btn(btn_row, "⏹  Parar",     C["danger"], C["text"], self.parar         ).pack(side="left", padx=4)
        make_btn(btn_row, "⚙  Filtro",    C["card"],   C["text"], self.abrir_config_filtro).pack(side="left", padx=4)

        # Checkboxes
        chk_row = tk.Frame(ctrl, bg=C["bg"])
        chk_row.pack(side="left", padx=16)

        for texto, var in [("Tracking", self.modo_tracking),
                            ("Filtrar classes", self.filtro_ativo)]:
            tk.Checkbutton(chk_row, text=texto, variable=var,
                           bg=C["bg"], fg=C["text"],
                           selectcolor=C["card"],
                           font=("Segoe UI", 9),
                           activebackground=C["bg"],
                           activeforeground=C["accent"]).pack(side="left", padx=8)

    # ══════════════════════════════════════════════════════════
    # ABA 2 – PIPELINE
    # ══════════════════════════════════════════════════════════

    # Mapeamento etapa → (chave_dict, rótulo, grupo)
    ETAPAS_PIPELINE = [
        # Etapa 1
        ("original",  "Etapa 1 – Original",              "Etapa 1: Aquisição"),
        # Etapa 2
        ("gray",      "Etapa 2 – Escala de Cinza",        "Etapa 2: Processamento"),
        ("blur",      "Etapa 2 – Desfoque Gaussiano",     "Etapa 2: Processamento"),
        ("edges",     "Etapa 2 – Bordas (Canny)",         "Etapa 2: Processamento"),
        # Etapa 3
        ("hsv",       "Etapa 3 – HSV Modificado",         "Etapa 3: Análise de Cor"),
        ("canal_h",   "Etapa 3 – Canal H (Matiz)",        "Etapa 3: Análise de Cor"),
        ("canal_s",   "Etapa 3 – Canal S (Saturação)",    "Etapa 3: Análise de Cor"),
        ("canal_v",   "Etapa 3 – Canal V (Valor)",        "Etapa 3: Análise de Cor"),
        # Etapa 5
        ("thresh",    "Etapa 5 – Binarização (Otsu)",     "Etapa 5: Binarização"),
    ]

    def _construir_aba_pipeline(self, parent):
        # Cabeçalho da aba
        tk.Label(parent,
                 text="Visualização das etapas de processamento aplicadas a cada frame.",
                 bg=C["bg"], fg=C["sub"], font=("Segoe UI", 9)).pack(
                     anchor="w", padx=14, pady=(10, 4))

        # ── Scroll para acomodar tudo ──────────────────────────
        outer = tk.Frame(parent, bg=C["bg"])
        outer.pack(fill="both", expand=True, padx=10, pady=(0, 6))

        scroll_canvas = tk.Canvas(outer, bg=C["bg"], highlightthickness=0)
        scrollbar     = ttk.Scrollbar(outer, orient="vertical",
                                      command=scroll_canvas.yview)
        self._pip_inner = tk.Frame(scroll_canvas, bg=C["bg"])
        self._pip_inner.bind(
            "<Configure>",
            lambda e: scroll_canvas.configure(
                scrollregion=scroll_canvas.bbox("all")))

        scroll_canvas.create_window((0, 0), window=self._pip_inner, anchor="nw")
        scroll_canvas.configure(yscrollcommand=scrollbar.set)
        scroll_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self._pip_labels : dict[str, tk.Label] = {}  # chave → Label imagem
        self._pip_tkimgs : dict[str, ImageTk.PhotoImage] = {}

        # Agrupar etapas para exibição com título de seção
        grupos_vistos = []
        grupo_frames  = {}

        for chave, rotulo, grupo in self.ETAPAS_PIPELINE:
            if grupo not in grupos_vistos:
                grupos_vistos.append(grupo)
                # Cabeçalho do grupo
                g_hdr = tk.Frame(self._pip_inner, bg=C["bg"])
                g_hdr.pack(fill="x", padx=8, pady=(12, 4))
                tk.Label(g_hdr, text=grupo, bg=C["bg"], fg=C["accent"],
                         font=("Segoe UI", 10, "bold")).pack(side="left")
                tk.Frame(g_hdr, bg=C["border"], height=1).pack(
                    side="left", fill="x", expand=True, padx=(8, 0), pady=5)

                # Frame de cards do grupo
                g_cards = tk.Frame(self._pip_inner, bg=C["bg"])
                g_cards.pack(fill="x", padx=8)
                grupo_frames[grupo] = (g_cards, 0)  # (frame, col_count)

            g_frame, col = grupo_frames[grupo]

            # Card individual
            card = make_frame(g_frame, bg=C["card"],
                              border_color=C["border"], bd=1)
            card.grid(row=0, column=col, padx=6, pady=4, sticky="n")

            tk.Label(card, text=rotulo, bg=C["card"], fg=C["sub"],
                     font=("Segoe UI", 8, "bold")).pack(pady=(8, 4))

            lbl_img = tk.Label(card, bg=C["card"])
            lbl_img.pack(padx=6, pady=(0, 8))
            self._pip_labels[chave] = lbl_img

            grupo_frames[grupo] = (g_frame, col + 1)

        # ── Etapa 4 – Histograma (seção separada) ──────────────
        hist_hdr = tk.Frame(self._pip_inner, bg=C["bg"])
        hist_hdr.pack(fill="x", padx=8, pady=(12, 4))
        tk.Label(hist_hdr, text="Etapa 4: Histograma de Intensidade",
                 bg=C["bg"], fg=C["accent"],
                 font=("Segoe UI", 10, "bold")).pack(side="left")
        tk.Frame(hist_hdr, bg=C["border"], height=1).pack(
            side="left", fill="x", expand=True, padx=(8, 0), pady=5)

        hist_card = make_frame(self._pip_inner, bg=C["card"],
                               border_color=C["border"], bd=1)
        hist_card.pack(fill="x", padx=14, pady=(0, 8))

        self.fig, self.ax = plt.subplots(figsize=(9, 2.4), facecolor=C["card"])
        self.fig.subplots_adjust(left=0.06, right=0.98, top=0.85, bottom=0.2)
        self.canvas_hist = FigureCanvasTkAgg(self.fig, master=hist_card)
        self.canvas_hist.get_tk_widget().pack(padx=10, pady=(10, 4), fill="x")

        self.lbl_hist_info = tk.Label(hist_card,
                                      text="Threshold Otsu: —  |  Iluminação: —",
                                      bg=C["card"], fg=C["accent2"],
                                      font=("Segoe UI", 9))
        self.lbl_hist_info.pack(pady=(0, 10))

        # Dica
        tk.Label(parent,
                 text="Os dados acima são atualizados automaticamente a cada frame processado.",
                 bg=C["bg"], fg=C["sub"], font=("Segoe UI", 7)).pack(pady=(0, 6))

    # ──────────────────────────────────────────────────────────
    # PLACEHOLDER
    # ──────────────────────────────────────────────────────────
    def _placeholder(self):
        c = self.canvas_video
        c.delete("all")
        c.create_rectangle(0, 0, 640, 480, fill=C["inactive"], outline="")
        for x in range(0, 641, 80):
            c.create_line(x, 0, x, 480, fill=C["card"], width=1)
        for y in range(0, 481, 60):
            c.create_line(0, y, 640, y, fill=C["card"], width=1)
        c.create_rectangle(1, 1, 639, 479, outline=C["border"], width=1)
        c.create_text(320, 200, text="AnalizeIA",
                      font=("Segoe UI", 28, "bold"), fill=C["accent2"])
        c.create_text(320, 250, text="Clique em  ▶ Webcam  ou  📁 Imagem",
                      font=("Segoe UI", 11), fill=C["sub"])
        c.create_text(320, 460, text="YOLOv8n · OpenCV · Python",
                      font=("Segoe UI", 8), fill=C["border"])

    # ──────────────────────────────────────────────────────────
    # CONTROLES
    # ──────────────────────────────────────────────────────────
    def iniciar_webcam(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("AnalizeIA", "Não foi possível acessar a webcam.")
            return
        self.running     = True
        self.frame_count = 0
        self.total_objetos = 0
        self.t0          = time.time()
        threading.Thread(target=self._loop_deteccao, daemon=True).start()

    def abrir_imagem(self):
        path = filedialog.askopenfilename(
            filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp *.webp")])
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("AnalizeIA", "Não foi possível carregar a imagem.")
            return
        self._processar_frame(img, fps=None)

    def parar(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.root.after(0, self.lbl_status.config,
                        {"text": "● Parado", "fg": C["sub"]})

    def abrir_config_filtro(self):
        JanelaFiltro(self.root, self.classes_filtro,
                     lambda s: setattr(self, "classes_filtro", s))

    # ──────────────────────────────────────────────────────────
    # LOOP DE CAPTURA
    # ──────────────────────────────────────────────────────────
    def _loop_deteccao(self):
        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            self.frame_count += 1
            elapsed = time.time() - self.t0
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            self._processar_frame(frame, fps=fps)
        self.running = False

    # ──────────────────────────────────────────────────────────
    # PROCESSAMENTO CENTRAL
    # ──────────────────────────────────────────────────────────
    def _processar_frame(self, frame: np.ndarray, fps: float | None):
        conf = float(self.slider_conf.get())

        # Etapas 1–5: Pipeline de imagem
        pipeline_data        = processar_pipeline(frame)
        self._ultimo_pipeline = pipeline_data
        thresh_val           = pipeline_data.get("thresh_val", 0)

        # Etapa 6: YOLO
        if self.modo_tracking.get():
            results = self.model.track(frame, persist=True, conf=conf)
        else:
            results = self.model(frame, conf=conf, verbose=False)

        annotated = results[0].plot()

        # Classes
        cls_ids   = results[0].boxes.cls.cpu().numpy().astype(int) \
                    if len(results[0].boxes) else []
        cls_nomes = [self.model.names[c] for c in cls_ids]

        # Filtro de classes (Desafio 4)
        if self.filtro_ativo.get() and self.classes_filtro:
            idx_ok = [i for i, n in enumerate(cls_nomes) if n in self.classes_filtro]
            if len(idx_ok) < len(cls_nomes):
                annotated = frame.copy()
                for i in idx_ok:
                    box = results[0].boxes.xyxy[i].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = box
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (180, 180, 180), 2)
                    cv2.putText(annotated, cls_nomes[i], (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)
            cls_nomes = [n for n in cls_nomes if n in self.classes_filtro]

        # Tracking IDs (Desafio 3)
        if self.modo_tracking.get() and results[0].boxes.id is not None:
            ids = results[0].boxes.id.cpu().numpy()
            for i, obj_id in enumerate(ids):
                if i >= len(results[0].boxes.xyxy):
                    continue
                box = results[0].boxes.xyxy[i].cpu().numpy().astype(int)
                x1, y1 = box[0], box[1]
                cv2.putText(annotated, f"ID:{int(obj_id)}", (x1, y1 - 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 2)

        contagem = len(cls_nomes)
        self.total_objetos += contagem

        # Overlays no vídeo (Etapa 7)
        if contagem > ALERT_THRESHOLD:
            cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 40), (60, 20, 20), -1)
            cv2.putText(annotated, f"ALERTA: {contagem} objetos detectados", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 80, 80), 2)

        cv2.putText(annotated, f"Obj: {contagem}", (10, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)
        if fps:
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 94),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 2)
        cv2.putText(annotated, f"Otsu: {thresh_val:.0f}", (10, 116),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (130, 130, 130), 2)

        # Preparar frame para exibição
        img_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        h_, w_  = img_rgb.shape[:2]
        scale   = min(640/w_, 480/h_, 1.0)
        img_rgb = cv2.resize(img_rgb, (int(w_*scale), int(h_*scale)))
        img_tk  = ImageTk.PhotoImage(Image.fromarray(img_rgb))

        # Agendar atualização na thread principal
        self.root.after(0, self._atualizar_ui,
                        img_tk, contagem, fps, list(set(cls_nomes)),
                        thresh_val, pipeline_data)

    # ──────────────────────────────────────────────────────────
    # ATUALIZAÇÃO DA UI (thread principal)
    # ──────────────────────────────────────────────────────────
    def _atualizar_ui(self, img_tk, contagem, fps,
                      classes_unicas, thresh_val, pipeline_data):

        # Aba 1 – vídeo
        self.canvas_video.delete("all")
        self.canvas_video.imgtk = img_tk
        self.canvas_video.create_image(0, 0, anchor="nw", image=img_tk)

        self.lbl_fps.config(text=f"{fps:.1f}" if fps else "—")
        self.lbl_frame_objs.config(
            text=str(contagem),
            fg=C["danger"] if contagem > ALERT_THRESHOLD else C["accent"])
        self.lbl_total.config(text=str(self.total_objetos))
        self.lbl_thresh.config(text=f"{thresh_val:.0f}")

        if contagem > ALERT_THRESHOLD:
            self.lbl_status.config(text="⚠ Alerta!", fg=C["danger"])
        elif contagem > 0:
            self.lbl_status.config(text="● Detectando", fg=C["ok"])
        else:
            self.lbl_status.config(text="● Aguardando", fg=C["sub"])

        self.txt_classes.config(state="normal")
        self.txt_classes.delete("1.0", "end")
        for c in sorted(classes_unicas):
            self.txt_classes.insert("end", f"  {c}\n")
        self.txt_classes.config(state="disabled")

        # Aba 2 – pipeline (sempre atualiza, mesmo se a aba não estiver visível)
        self._atualizar_pipeline(pipeline_data, thresh_val)

    def _atualizar_pipeline(self, data: dict, thresh_val: float):
        # Thumbnails
        for chave, lbl in self._pip_labels.items():
            if chave in data:
                tk_img = thumb(data[chave])
                self._pip_tkimgs[chave] = tk_img
                lbl.config(image=tk_img)

        # Histograma
        hist = data.get("hist_data")
        if hist is not None:
            self.ax.clear()
            self.ax.set_facecolor(C["card"])
            self.fig.patch.set_facecolor(C["card"])
            self.ax.hist(hist, bins=256, range=(0, 256),
                         color=C["accent2"], alpha=0.9, linewidth=0)
            self.ax.set_xlim(0, 255)
            self.ax.tick_params(colors=C["sub"], labelsize=7)
            for spine in self.ax.spines.values():
                spine.set_edgecolor(C["border"])
            self.ax.set_xlabel("Intensidade (0–255)", color=C["sub"], fontsize=8)
            self.ax.set_ylabel("Freq.", color=C["sub"], fontsize=8)
            self.ax.set_title("Histograma de Intensidade", color=C["text"], fontsize=9)
            self.canvas_hist.draw()

            mean_i = float(np.mean(hist))
            if mean_i < 80:
                ilum = "Subexposta (escura)"
            elif mean_i > 180:
                ilum = "Superexposta (clara)"
            else:
                ilum = "Boa dinâmica tonal"
            self.lbl_hist_info.config(
                text=f"Threshold Otsu: {thresh_val:.0f}  |  Iluminação: {ilum}")


# ──────────────────────────────────────────────────────────────
# PONTO DE ENTRADA
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app  = AnalizeIA(root)
    root.mainloop()
