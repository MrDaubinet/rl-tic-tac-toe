<script>
	// store
  import history from "../stores/history.js"
  // store information
  import { game } from "../game.js"
  // function prop
  let { updateState } = $props();

  // Go back in game time
  /**
     * @param {number} index
     */
  async function goBack(index) {
    // remove states from history
    history.selectState(index)
    // update game state to the histories latest state
    game.setStates($history[$history.length - 1].state)
    // update game turn to the histories latest turn
    game.setTurn($history[$history.length - 1].turn)
    // update the board state
    updateState()
  }
</script>

{#each $history as state, index}
  {#if index != ($history.length -1)}
    <div class="{index == 0 ? 'pb-4' : 'py-4'}">
      <button type="button"
        onclick={() => goBack(index)}
        class="flex justify-center text-2xl cursor-pointer hover:underline w-full"
        >
        <h1> Go Back, </h1>
        <div class="ml-2">
          {#if index == 0}
            to start
          {:else}
            to move {index }
          {/if}
        </div>
      </button>
    </div>
  {/if}
{/each} 