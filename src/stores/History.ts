import { writable } from 'svelte/store';
export const history = writable([]);

export function addState(state, turn) {
  history.update(list => {
    list.push(
      {
        state: [...state],
        turn,
      }
    )
    return list
  });
}

export function resetState() {
  history.update(list => {
    list = []
    return list
  });
}

export function selectState(index) {
  history.update(list => {
    return list.slice(0, index);
    // list = [...list]
    // return list
  });
}


// export function addRobotMessage(message) {
//   messages.update(list => {
//     list.push(
//       {
//         message: message,
//         date: new Date(),
//         type: "bot"
//       }
//     )
//     return list
//   });
// }

// export function deleteMessage(index) {
//   messages.update(list => {
//     list.splice(index, 1)
//   });
// }

// export function refreshMessages(index) {
//   messages.set([
//     {
//       message: "My Memory has been cleared... 🤖 How can I assist?",
//       type: "bot"
//     }
//   ]);
// }