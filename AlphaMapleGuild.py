from datetime import datetime
import cv2
import numpy as np
import openpyxl
import time
from PIL import Image as im
from PIL import ImageGrab
import os


template = cv2.imread(r"data\guild.png", cv2.IMREAD_GRAYSCALE)
mask = cv2.imread(r"data\mask.png", cv2.IMREAD_GRAYSCALE)
numpic = cv2.imread(r"data\maple_numbers.png", cv2.IMREAD_GRAYSCALE) * 255
thousand = cv2.imread(r"data\maple_1000.png", cv2.IMREAD_GRAYSCALE)

nums = [numpic[:, 7 * i:7 * i + 7] for i in range(11)]

saved_previous = None

mDict = {}

patience = 10
total = 0
while True:
    start_time = time.time()
    color = np.array(ImageGrab.grab(all_screens=True))
    print(r"화면 캡쳐 완료! 처리중...", end='')
    twobit = np.prod(color == 255, axis=-1) + np.prod(color == 179, axis=-1) + np.prod(color == 187, axis=-1)

    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(gray, template, cv2.TM_SQDIFF_NORMED, mask=mask)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
    x, y = minLoc
    h, w = template.shape

    cut = np.array(twobit)[y: y + h, x: x + w][131:539, 205:640]

    missions = [np.argmax([np.prod(np.equal(cut[x:x + 10, 273:280], num))
                           for num in nums[:6]])
                for x in range(8, cut.shape[0], 24)]

    suro = [np.argmax([np.prod(np.equal(cut[x:x + 10, 327:334], num))
                       for num in nums[:10]]) * 1000 +
            np.argmax([np.prod(np.equal(cut[x:x + 10, 336:343], num))
                       for num in nums[:10]]) * 100 +
            np.argmax([np.prod(np.equal(cut[x:x + 10, 342:349], num))
                       for num in nums[:10]]) * 10 +
            np.argmax([np.prod(np.equal(cut[x:x + 10, 348:355], num))
                       for num in nums[:10]]) +
            np.argmax([np.prod(np.equal(cut[x:x + 10, 331:338], num))
                       for num in nums[:10]]) * 100 +
            np.argmax([np.prod(np.equal(cut[x:x + 10, 337:344], num))
                       for num in nums[:10]]) * 10 +
            np.argmax([np.prod(np.equal(cut[x:x + 10, 343:350], num))
                       for num in nums[:10]])
            for x in range(8, cut.shape[0], 24)]

    flag = [np.prod(np.equal(cut[x:x + 10, 407:433] * 255, thousand)) * 1000 +
            np.argmax([np.prod(np.equal(cut[x:x + 10, 409:416], num))
                       for num in nums[:10]]) * 100 +
            np.argmax([np.prod(np.equal(cut[x:x + 10, 415:422], num))
                       for num in nums[:10]]) * 10 +
            np.argmax([np.prod(np.equal(cut[x:x + 10, 421:428], num))
                       for num in nums[:10]])
            for x in range(8, cut.shape[0], 24)]

    nick_img = []
    nick_hash = []
    for x in range(7, cut.shape[0] + 25, 24):
        for y in range(8, 73):
            if np.sum(cut[x + 1:x + 11, y:y + 4]) == 0:
                break
        nick = cut[x:x + 12, 7:y + 1]
        nick_img.append(nick)
        nick_hash.append(hex(y - 6) + ''.join([hex(i % 16)[2:] for i in np.sum(nick[1:-1], axis=1)]))

    if not os.path.exists('nicknames'):
        os.makedirs('nicknames')

    cnt = 0
    for nick, a, b, c, d in zip(nick_hash, nick_img, missions, suro, flag):
        if nick != '0x20000000000' and nick not in mDict:
            im.fromarray(255 - a * 255).convert('RGB').save(f'nicknames/{nick}.jpg')
            mDict[nick] = (b, c, d)
            cnt += 1
    if cnt:
        total += cnt
        patience = 5
        print(f"\r{cnt}명 추가, 누적 {total}명 기록")
    else:
        patience -= 1
        if patience <= 0:
            print('\r캡처를 종료합니다')
            break
        else:
            print(f"\r스크롤을 넘겨주세요. {patience}초 더 대기하면 프로그램이 종료됩니다.")
    time.sleep(max(1 - time.time() + start_time, 0))

hash2nick = {}

try:
    wb = openpyxl.load_workbook('result.xlsx')
    w0 = wb.worksheets[0]
    w0.column_dimensions['B'].hidden = True

except FileNotFoundError:
    wb = openpyxl.Workbook()
    w0 = wb.worksheets[0]
    w0.title = "닉네임 정보"
    w0.cell(row=1, column=1).value = '닉네임 이미지'
    w0.cell(row=1, column=2).value = '닉네임 해쉬'
    w0.cell(row=1, column=3).value = '닉네임(수동입력)'

ws = wb.create_sheet(title=datetime.now().strftime("%y%m%d %H%M"))
ws.cell(row=1, column=1).value = '닉네임 해쉬'
ws.cell(row=1, column=2).value = '닉네임(수동입력)'
ws.cell(row=1, column=3).value = '주간미션'
ws.cell(row=1, column=4).value = '지하 수로'
ws.cell(row=1, column=5).value = '플래그 레이스'

for i, (h, (m, s, f)) in enumerate(mDict.items()):
    ws.cell(row=i + 2, column=1).value = h
    if h in hash2nick:
        ws.cell(row=i + 2, column=3).value = hash2nick[h]
    else:
        max_r = w0.max_row
        img = openpyxl.drawing.image.Image(f'nicknames/{h}.jpg')
        img.anchor = f'A{max_r + 1}'
        w0.add_image(img)
        w0.cell(row=max_r + 1, column=2).value = h

    ws.cell(row=i + 2, column=2).value = f"=IF(VLOOKUP(A{i + 2},'닉네임 정보'!B:C,2,FALSE)" + r',,"")'
    ws.cell(row=i + 2, column=3).value = m
    ws.cell(row=i + 2, column=4).value = s
    ws.cell(row=i + 2, column=5).value = f

while True:
    try:
        wb.save('result.xlsx')
        break
    except PermissionError:
        print("권한 오류 발생, 1초 뒤 재시도합니다. 주로 result.xlsx가 열려 있는 경우 발생합니다.")
        time.sleep(1)
