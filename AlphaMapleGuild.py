from datetime import datetime
import cv2
import numpy as np
import openpyxl
import time
from PIL import ImageGrab
import os
import hashlib

GUILD_TEMPLATE = cv2.imread(r"data\guild.png", cv2.IMREAD_GRAYSCALE)
GUILD_MASK = cv2.imread(r"data\mask.png", cv2.IMREAD_GRAYSCALE)
GUILD_TITLE = cv2.imread(r"data\guild_member_participation.png", cv2.IMREAD_GRAYSCALE)
NUMBER_PICTURES = cv2.imread(r"data\maple_numbers.png", cv2.IMREAD_GRAYSCALE) * 255
NUMBERS = cv2.imread(r"data\numbers.png", cv2.IMREAD_GRAYSCALE).reshape(8, 10, 5).swapaxes(0, 1) * 255
COMMA = cv2.imread(r"data\comma.png", cv2.IMREAD_GRAYSCALE) * 255
THOUSAND_PICTURE = cv2.imread(r"data\number_1000.png", cv2.IMREAD_GRAYSCALE) * 255
GUILD_H, GUILD_W = GUILD_TEMPLATE.shape

saved_previous = None

written_nicknames = {}

TITLE_THRESHOLD = 10000
patience = 10
total = 0

print("전체 화면을 캡처해 길드 창의 위치를 찾습니다. 모니터의 해상도가 높을수록 오래 걸릴 수 있습니다.")


capture = np.array(ImageGrab.grab(all_screens=True))
capture = cv2.cvtColor(capture, cv2.COLOR_BGR2GRAY)
match_result = cv2.minMaxLoc(cv2.matchTemplate(capture, GUILD_TEMPLATE, cv2.TM_SQDIFF_NORMED, mask=GUILD_MASK))
best_error, worst_error, best_location, worst_location = match_result

print(f"이미지 오차 점수: {best_error:.3f}, 좌상단 픽셀 좌표 (x, y): {best_location}")
print("(정상적인 경우 오차는 대체로 0.01 이하입니다)")

guild_x, guild_y = best_location
capture = capture[guild_y:, guild_x:]

while np.sum(GUILD_TITLE - capture[64:83, 175:286]) > TITLE_THRESHOLD:
    inputs = input("현재 탭이 [길드원 참여 현황]이 아닙니다. 이 창에서 엔터 키를 입력해 재확인합니다. SKIP을 입력해 무시할 수 있습니다.")
    if inputs == 'SKIP':
        break
    capture = np.array(ImageGrab.grab(bbox=(guild_x, guild_y, guild_x + GUILD_W, guild_y + GUILD_H), all_screens=True))
    capture = cv2.cvtColor(capture, cv2.COLOR_BGR2GRAY)
print("현재 길드 창의 탭이 [길드원 참여 현황]임을 확인하였습니다.")

start_time = time.time()
while True:
    x = guild_x + 205
    y = guild_y + 131
    w = guild_x + 640
    h = guild_y + 539
    capture = np.array(ImageGrab.grab(bbox=(x, y, w, h), all_screens=True))
    capture = np.prod(capture == 255, axis=-1) + np.prod(capture == 179, axis=-1) + np.prod(capture == 187, axis=-1)
    capture = capture.astype('u1')
    # grayscale color == (255 or 179 or 187)
    cv2.imwrite(f'tmp.png', 255 - capture * 255)

    # capture.shape = (H, W)

    mission_score = np.zeros(17, dtype=int)
    idx, val = np.where(np.all(capture[:, 274:279].reshape(17, 24, -1)[np.newaxis, :, 9:17] == NUMBERS[:6, np.newaxis, ...], axis=(2, 3)).T)
    mission_score[idx] = val

    flag_score = np.all(capture[:, 408:432].reshape(17, 24, -1)[:, 9:18] == THOUSAND_PICTURE[np.newaxis, :, :], axis=(1, 2)) * 1000
    idx, val = np.where(np.all(capture[:, 410:415].reshape(17, 24, -1)[np.newaxis, :, 9:17] == NUMBERS[:, np.newaxis, ...], axis=(2, 3)).T)
    flag_score[idx] += val * 100
    idx, val = np.where(np.all(capture[:, 416:421].reshape(17, 24, -1)[np.newaxis, :, 9:17] == NUMBERS[:, np.newaxis, ...], axis=(2, 3)).T)
    flag_score[idx] += val * 10

    def detect_number(start):
        return np.where(np.all(capture[:, start:start+5].reshape(17, 24, -1)[np.newaxis, :, 9:17] == NUMBERS[:, np.newaxis, ...], axis=(2, 3)).T)

    suro_score = np.zeros(17, dtype=int)

    idx, val = detect_number(325)
    suro_score[idx] += val * 10000
    idx, val = detect_number(331)
    suro_score[idx] += val * 1000
    idx, val = detect_number(340)
    suro_score[idx] += val * 100
    idx, val = detect_number(346)
    suro_score[idx] += val * 10
    idx, val = detect_number(352)
    suro_score[idx] += val * 1

    idx, val = detect_number(328)
    suro_score[idx] += val * 1000
    idx, val = detect_number(337)
    suro_score[idx] += val * 100
    idx, val = detect_number(343)
    suro_score[idx] += val * 10
    idx, val = detect_number(349)
    suro_score[idx] += val * 1

    idx, val = detect_number(332)
    suro_score[idx] += val * 100
    idx, val = detect_number(338)
    suro_score[idx] += val * 10
    idx, val = detect_number(333)
    suro_score[idx] += val * 1

    nickname_imgs = capture[:, 7:76].reshape(17, 24, -1)[:, 8:19]

    if not os.path.exists('nick_hash'):
        os.makedirs('nick_hash')
    blank_hash = "2269c8cf4be7a9ed867972bcd37b73c1ed5a6e97212c3717a804b16c9795b8df"

    cnt = 0
    for img, mission, suro, flag in zip(nickname_imgs, mission_score, suro_score, flag_score):
        nick_hash = hashlib.sha256(str(img.tobytes()).encode()).hexdigest()
        if nick_hash != blank_hash and nick_hash not in written_nicknames:
            cv2.imwrite(f'nick_hash/{nick_hash}.png', 255 - img * 255)
            written_nicknames[nick_hash] = (mission, suro, flag)
            cnt += 1

    if cnt:
        total += cnt
        print(f"\r{cnt}명 추가, 누적 {total}명 기록")
        start_time = time.time()
    elif (time.time() - start_time) > 5 or total > 188:
        print('\r시간이 초과되었거나, 모든 페이지를 스캔하여 스캔을 종료합니다')
        break
    else:
        print(f"\r스크롤을 넘겨주세요. 새로운 페이지가 5초간 스캔되지 않으면 스캔을 종료합니다.", end='')

try:
    wb = openpyxl.load_workbook('result.xlsx')
    w0 = wb.worksheets[0]
    hash2nick = {r[0]: r[1]
                 for r in w0.iter_rows(min_row=2, max_row=w0.max_row, min_col=2, max_col=3, values_only=True)}
except FileNotFoundError:
    wb = openpyxl.Workbook()
    w0 = wb.worksheets[0]
    w0.title = "닉네임 정보"
    w0.column_dimensions['B'].hidden = True
    w0.cell(row=1, column=1).value = '닉네임 이미지'
    w0.cell(row=1, column=2).value = '닉네임 해쉬'
    w0.cell(row=1, column=3).value = '닉네임(수동입력)'
    hash2nick = {}

ws = wb.create_sheet(title=datetime.now().strftime("%y%m%d %H%M"))
ws.cell(row=1, column=1).value = '닉네임 해쉬'
ws.cell(row=1, column=2).value = '닉네임(자동으로 채워짐)'
ws.cell(row=1, column=3).value = '주간미션'
ws.cell(row=1, column=4).value = '지하 수로'
ws.cell(row=1, column=5).value = '플래그 레이스'

num_new_nickname = 0
for i, (h, (m, s, f)) in enumerate(written_nicknames.items()):
    ws.cell(row=i + 2, column=1).value = h
    if h in hash2nick and hash2nick[h] != '':
        ws.cell(row=i + 2, column=2).value = hash2nick[h]
        del hash2nick[h]
    else:
        ws.cell(row=i + 2, column=2).value = f"=IF(VLOOKUP(A{i + 2},'닉네임 정보'!B:C,2,FALSE)" + r',,"")'
        num_new_nickname += 1
        max_r = w0.max_row
        img = openpyxl.drawing.image.Image(f'nick_hash/{h}.png')
        img.anchor = f'A{max_r + 1}'
        w0.add_image(img)
        w0.cell(row=max_r + 1, column=2).value = h

    ws.cell(row=i + 2, column=3).value = m
    ws.cell(row=i + 2, column=4).value = s
    ws.cell(row=i + 2, column=5).value = f

flag = False
if len(hash2nick) > 0:
    print(f"기존에 존재한 닉네임 중 총 {len(hash2nick)}개가 스캔되지 않았습니다.")
    print(f"스캔 도중 마우스가 닉네임을 가렸거나, 기존 유저가 닉네임을 변경했거나, 길드를 탈퇴하였을 수 있습니다.")
    print("누락된 닉네임 중, 수기로 기록되어 있던 닉네임은 다음과 같습니다.")
    print([v for v in list(hash2nick.values()) if v])
    flag = True
if num_new_nickname > 0:
    print(f"새로 추가된 닉네임은 총 {num_new_nickname}개 입니다.")
    flag = True
if flag:
    input("닉네임이 누락되")

while True:
    try:
        wb.save('result.xlsx')
        break
    except PermissionError:
        print("권한 오류 발생, 1초 뒤 재시도합니다. 주로 result.xlsx가 열려 있는 경우 발생합니다.")
        time.sleep(1)
