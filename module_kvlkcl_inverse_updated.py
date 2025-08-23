import requests, importlib.util, sys
from IPython.display import display

# --- دانلود و بارگذاری ماژول از گیت‌هاب ---
url = "https://raw.githubusercontent.com/Lab2Phys/kvlkcl-inverse-module-/refs/heads/main/module_kvlkcl_inverse_updated.py"
module_filename = "module_kvlkcl_inverse_updated.py"
try:
    open(module_filename, "wb").write(requests.get(url).content)
    print("✅ Module downloaded successfully.")
except Exception as e:
    print(f"❌ Failed to download module: {e}")

# --- بارگذاری ماژول در محیط پایتون ---
spec = importlib.util.spec_from_file_location("module_kvlkcl_inverse", module_filename)
m = importlib.util.module_from_spec(spec)
sys.modules["module_kvlkcl_inverse"] = m
spec.loader.exec_module(m)
create_analyzer_ui = m.create_analyzer_ui

# ==============================================================================
# ⚙️ بخش تنظیمات مدار - اینجا را ویرایش کنید
# ==============================================================================

# مشکل دوم: تعیین تعداد ارقام اعشار برای نمایش خروجی
DECIMAL_PLACES = 4 

# مقادیر ثابت مقاومت و منابع ولتاژ معلوم
R = 1000
e2 = 5
e3 = 12

# تعریف یال‌ها (شاخه‌ها) به صورت: (گره اول, گره دوم, مقاومت)
edges = [
    (1, 2, R), (1, 3, 2*R), (1, 6, 2*R), (2, 4, R), (2, 5, R),
    (2, 7, 0), (3, 4, R), (3, 8, R), (4, 7, R), (5, 6, R),
    (5, 8, R), (6, 8, R), (7, 8, R)
]

# تعریف منابع ولتاژ به صورت: {(گره اول, گره دوم): مقدار}
# برای منبع مجهول، از ضریب 1+ (برای هم‌جهت با مسیر) یا 1- (برای خلاف جهت) استفاده کنید.
# جهت: از گره اول به دوم.
voltage_source_config = {
    (1, 3): -1,  # منبع مجهول e1 با ضریب 1-
    (1, 6): -e2,
    (2, 7): -e3
}

# تعریف حلقه‌های مستقل مدار
# جهت هر حلقه با ترتیب گره‌ها مشخص می‌شود.
loops = [
    [1, 2, 4, 3], [1, 2, 5, 6], [3, 4, 7, 8],
    [2, 4, 7], [2, 5, 8, 7], [5, 6, 8]
]

# مشکل سوم: تعیین تعداد گره‌ها و حلقه‌ها (این مقادیر به صورت خودکار محاسبه می‌شوند)
num_loops = len(loops)
# محاسبه تعداد گره‌های منحصر به فرد از لیست یال‌ها
num_nodes = len(set(n for edge in edges for n in edge[:2])) 
print(f"Circuit details: {num_nodes} nodes, {num_loops} independent loops.")


# ==============================================================================
# 🚀 اجرای تحلیل‌گر
# ==============================================================================

# ساخت و نمایش رابط کاربری با تنظیمات بالا
# تمام قابلیت‌های جدید (مشکلات ۲ تا ۶) در این تابع پیاده‌سازی شده‌اند.
ui = create_analyzer_ui(
    loops=loops,
    edges=edges,
    voltage_source_config=voltage_source_config,
    decimal_places=DECIMAL_PLACES
)

display(ui)